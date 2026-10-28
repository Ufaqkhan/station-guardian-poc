"""Background lock for a fixed camera: keep the generated people (and their shadows), take everything else from the
plate, so the platform does not drift across 18 chained 5-second generations.

Foreground mask per frame = pixels that differ from the plate (people, shadows, anything that moved), cleaned with
morphology, kept only near large changed regions, dilated, feathered in space and smoothed over +-2 frames.
If a frame's background has drifted globally (the mask would cover most of the frame), the lock is skipped for
that frame and reported.

  python scripts/bglock.py --clip take.mkv --plate plate.png --out take_locked.mkv [--debug]
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from assemble import read_frames  # noqa: E402


def write_ffv1(frames, path, fps=16):
    n, h, w, _ = frames.shape
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                    "-r", str(fps), "-i", "-", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", path],
                   input=np.ascontiguousarray(frames).tobytes(), check=True)


def fg_masks(frames, plate, thr=16.0, min_area=600, dilate=9, feather=5.0):
    pl = cv2.GaussianBlur(cv2.cvtColor(plate, cv2.COLOR_RGB2LAB).astype(np.float32), (0, 0), 1.5)
    masks = []
    for f in frames:
        lab = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2LAB).astype(np.float32), (0, 0), 1.5)
        d = np.sqrt(((lab - pl) ** 2 * np.array([0.6, 1.0, 1.0], np.float32)).sum(-1))
        m = (d > thr).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[lab_cc == i] = 1
        # fill holes inside kept regions
        cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(keep)
        cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
        filled = cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)))
        masks.append(filled.astype(np.float32))
    masks = np.stack(masks)
    # temporal max over +-2 frames (no popping at the mask edge), then spatial feather
    tm = masks.copy()
    for k in (1, 2):
        tm[k:] = np.maximum(tm[k:], masks[:-k])
        tm[:-k] = np.maximum(tm[:-k], masks[k:])
    return np.stack([cv2.GaussianBlur(m, (0, 0), feather) for m in tm]).clip(0, 1)


def stabilize(frames, plate, scale=0.5):
    """Align every frame to the plate with an affine fitted on background pixels only (ECC, half resolution).

    The generator's camera wobbles by 1-3 px and breathes by <0.5 %; a bolted CCTV camera does not.
    Foreground (people, shadows) is excluded from the fit with a generous difference mask.
    """
    H, W = plate.shape[:2]
    pg = cv2.cvtColor(plate.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    ps = cv2.resize(pg, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
    rough = fg_masks(frames, plate, thr=22.0, dilate=25, feather=0.01) > 0.5
    out, warps = np.empty_like(frames), []
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    for i, f in enumerate(frames):
        fs = cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA).astype(np.float32)
        bgm = cv2.resize((~rough[i]).astype(np.uint8), (fs.shape[1], fs.shape[0]), interpolation=cv2.INTER_NEAREST)
        try:
            _, warp = cv2.findTransformECC(ps, fs, warp.copy(), cv2.MOTION_AFFINE, crit, bgm, 5)
        except cv2.error:
            pass  # keep the previous frame's warp
        w = warp.copy()
        w[:, 2] /= scale
        warps.append(w)
        out[i] = cv2.warpAffine(f, w, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REFLECT)
    return out, np.stack(warps)


def stabilize_h(frames, plate, refine=True):
    """Robust version: per-frame homography to the plate from SIFT matches + RANSAC (people are outliers),
    refined with ECC on background pixels. A pan/zoom of a fixed camera is exactly a homography."""
    H_, W_ = plate.shape[:2]
    sift = cv2.SIFT_create(nfeatures=5000)
    pg = cv2.cvtColor(plate.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    kp_p, des_p = sift.detectAndCompute(pg, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    out, Hs, inl = np.empty_like(frames), [], []
    Hprev = np.eye(3)
    ps = cv2.resize(pg, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32)
    for i, f in enumerate(frames):
        fg = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        kp_f, des_f = sift.detectAndCompute(fg, None)
        H = Hprev
        if des_f is not None and len(kp_f) > 20:
            m = [a for a, b in bf.knnMatch(des_f, des_p, k=2) if a.distance < 0.75 * b.distance]
            if len(m) >= 20:
                src = np.float32([kp_f[a.queryIdx].pt for a in m])
                dst = np.float32([kp_p[a.trainIdx].pt for a in m])
                Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
                if Hm is not None and mask.sum() >= 15:
                    H = Hm
                    inl.append(int(mask.sum()))
        w = cv2.warpPerspective(f, H, (W_, H_), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        if refine:
            # ECC refinement on background pixels of the aligned frame (half resolution)
            d = np.abs(cv2.cvtColor(w, cv2.COLOR_RGB2GRAY).astype(np.float32) - pg)
            bgm = (cv2.GaussianBlur(d, (0, 0), 3) < 18).astype(np.uint8)
            bgm = cv2.erode(bgm, np.ones((15, 15), np.uint8))
            ws = cv2.resize(cv2.cvtColor(w, cv2.COLOR_RGB2GRAY), None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_AREA).astype(np.float32)
            bs = cv2.resize(bgm, (ws.shape[1], ws.shape[0]), interpolation=cv2.INTER_NEAREST)
            E = np.eye(3, dtype=np.float32)
            try:
                _, E = cv2.findTransformECC(ps, ws, E, cv2.MOTION_HOMOGRAPHY,
                                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6), bs, 3)
                S = np.diag([2.0, 2.0, 1.0]).astype(np.float32)
                E = S @ E @ np.linalg.inv(S)          # half-res -> full-res
                if abs(E[0, 2]) < 6 and abs(E[1, 2]) < 6:
                    H = np.linalg.inv(E) @ H
                    w = cv2.warpPerspective(f, H, (W_, H_), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            except cv2.error:
                pass
        Hprev = H
        Hs.append(H)
        out[i] = w
    return out, np.stack(Hs), inl


def local_homographies(frames, plate, people, ring=260, min_inl=25, smooth=2):
    """Per-frame homography frame->plate fitted on the context around the people (bench, nearby ground, fence),
    not the people themselves. The generator's camera also translates a little, so far background and the bench
    near the people need different alignments (parallax); only the near one matters for the composite.
    Falls back to the whole-frame fit when the ring has too few matches. Smoothed over +-smooth frames
    (corner trajectories), so the people do not jitter."""
    Hh, Ww = plate.shape[:2]
    sift = cv2.SIFT_create(nfeatures=6000)
    pg = cv2.cvtColor(plate.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    kp_p, des_p = sift.detectAndCompute(pg, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    kr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    kp4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    Hs, stats = [], []
    Hprev = np.eye(3)
    for i, f in enumerate(frames):
        fg = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        kp_f, des_f = sift.detectAndCompute(fg, None)
        H, used = Hprev, "prev"
        if des_f is not None and len(kp_f) > 30:
            m = [a for a, b in bf.knnMatch(des_f, des_p, k=2) if a.distance < 0.75 * b.distance]
            src = np.float32([kp_f[a.queryIdx].pt for a in m])
            dst = np.float32([kp_p[a.trainIdx].pt for a in m])
            pm = people[i].astype(np.uint8)
            if pm.any():
                ringm = cv2.dilate(pm, kr) & (1 - cv2.dilate(pm, kp4))
                sel = ringm[src[:, 1].astype(int).clip(0, Hh - 1), src[:, 0].astype(int).clip(0, Ww - 1)] > 0
            else:
                sel = np.zeros(len(src), bool)
            Hl, ml = (cv2.findHomography(src[sel], dst[sel], cv2.RANSAC, 2.0) if sel.sum() >= min_inl
                      else (None, None))
            if Hl is not None and ml.sum() >= min_inl:
                H, used = Hl, f"ring{int(ml.sum())}"
            else:
                keep = ~people[i][src[:, 1].astype(int).clip(0, Hh - 1), src[:, 0].astype(int).clip(0, Ww - 1)]
                Hg, mg = cv2.findHomography(src[keep], dst[keep], cv2.RANSAC, 2.0) if keep.sum() >= 20 else (None, None)
                if Hg is not None:
                    H, used = Hg, f"global{int(mg.sum())}"
        Hprev = H
        Hs.append(H / H[2, 2])
        stats.append(used)
    # temporal smoothing on the image of the four corners
    C = np.float32([[0, 0], [Ww, 0], [Ww, Hh], [0, Hh]]).reshape(-1, 1, 2)
    P = np.stack([cv2.perspectiveTransform(C, H).reshape(4, 2) for H in Hs])
    Ps = P.copy()
    for i in range(len(P)):
        a, b = max(0, i - smooth), min(len(P), i + smooth + 1)
        Ps[i] = P[a:b].mean(0)
    Hs = np.stack([cv2.getPerspectiveTransform(C.reshape(4, 2), Ps[i].astype(np.float32)) for i in range(len(P))])
    return Hs, stats


def warp_all(frames, masks, Hs, plate_shape):
    Hh, Ww = plate_shape[:2]
    out = np.empty((len(frames), Hh, Ww, 3), np.uint8)
    mo = np.zeros((len(frames), Hh, Ww), bool)
    for i, (f, H) in enumerate(zip(frames, Hs)):
        out[i] = cv2.warpPerspective(f, H, (Ww, Hh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        mo[i] = cv2.warpPerspective(masks[i].astype(np.uint8), H, (Ww, Hh), flags=cv2.INTER_NEAREST) > 0
    return out, mo


def lock(frames, plate, max_cover=0.45, stab=True, **kw):
    if stab:
        frames, warps, _ = stabilize_h(frames, plate)
    m = fg_masks(frames, plate, **kw)
    cover = m.reshape(len(m), -1).mean(1)
    out = frames.astype(np.float32).copy()
    skipped = 0
    for i in range(len(frames)):
        if cover[i] > max_cover:
            skipped += 1
            continue
        a = m[i][..., None]
        out[i] = a * frames[i] + (1 - a) * plate
    return out.round().clip(0, 255).astype(np.uint8), m, cover, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--plate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=16.0)
    ap.add_argument("--debug", action="store_true", help="also write the mask video next to the output")
    args = ap.parse_args()
    fr = read_frames(args.clip)
    plate = cv2.cvtColor(cv2.imread(args.plate), cv2.COLOR_BGR2RGB)
    if plate.shape[:2] != fr.shape[1:3]:
        plate = cv2.resize(plate, (fr.shape[2], fr.shape[1]), interpolation=cv2.INTER_AREA)
    out, m, cover, skipped = lock(fr, plate, thr=args.thr)
    write_ffv1(out, args.out)
    cv2.imwrite(os.path.splitext(args.out)[0] + "_last.png", cv2.cvtColor(out[-1], cv2.COLOR_RGB2BGR))
    if args.debug:
        vis = (np.concatenate([out, (m[..., None] * 255).repeat(3, -1).astype(np.uint8)], 2))
        write_ffv1(vis, os.path.splitext(args.out)[0] + "_mask.mkv")
    print(f"locked {args.clip}: mask cover mean {cover.mean():.3f} max {cover.max():.3f}, skipped {skipped} frames")


if __name__ == "__main__":
    main()


# ---- ground-plane registration (plate 1280x720 coordinates) ----
GROUND = [(95, 720), (458, 0), (780, 0), (780, 215), (840, 222), (845, 432), (1000, 432), (1055, 470), (1055, 655),
          (1235, 655), (1250, 720)]          # real GP02 view: open platform surface (benches, railing, bin excluded)
NOT_GROUND = []
GROUND_V1 = [(60, 720), (430, 0), (620, 0), (900, 268), (1200, 448), (1240, 720)]   # generated plate (v1)


def ground_mask(shape):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.array(GROUND, np.int32)], 1)
    for p in NOT_GROUND:
        cv2.fillPoly(m, [np.array(p, np.int32)], 0)
    return m


def ground_homographies(frames, plate, people, min_inl=30, smooth=2):
    """Per-frame homography frame->plate fitted only on matches that lie on the platform surface (a plane, so one
    homography is exact under any camera motion) and not on people. Smoothed over +-smooth frames."""
    Hh, Ww = plate.shape[:2]
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)
    pg = cv2.cvtColor(plate.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gm = ground_mask(plate.shape)
    kp_p, des_p = sift.detectAndCompute(pg, gm * 255)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    Hs, used = [], []
    Hprev = np.eye(3)
    for i, f in enumerate(frames):
        fg = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        notp = (1 - cv2.dilate(people[i].astype(np.uint8), kd)) * 255
        kp_f, des_f = sift.detectAndCompute(fg, notp)
        H, u = Hprev, "prev"
        if des_f is not None and len(kp_f) > 30:
            m = [a for a, b in bf.knnMatch(des_f, des_p, k=2) if a.distance < 0.8 * b.distance]
            if len(m) >= min_inl:
                src = np.float32([kp_f[a.queryIdx].pt for a in m])
                dst = np.float32([kp_p[a.trainIdx].pt for a in m])
                Hm, mk = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
                if Hm is not None and mk.sum() >= min_inl:
                    H, u = Hm, f"ground{int(mk.sum())}"
        Hprev = H
        Hs.append(H / H[2, 2])
        used.append(u)
    C = np.float32([[0, 0], [Ww, 0], [Ww, Hh], [0, Hh]]).reshape(-1, 1, 2)
    P = np.stack([cv2.perspectiveTransform(C, H).reshape(4, 2) for H in Hs])
    Ps = P.copy()
    for i in range(len(P)):
        a, b = max(0, i - smooth), min(len(P), i + smooth + 1)
        Ps[i] = P[a:b].mean(0)
    Hs = np.stack([cv2.getPerspectiveTransform(C.reshape(4, 2), Ps[i].astype(np.float32)) for i in range(len(P))])
    return Hs, used


def chained_homographies(frames, upto=None):
    """Frame t -> frame 0 homographies by chaining consecutive-frame fits (robust to large accumulated camera
    motion). Frame 0 of a take is the previous beat's locked last frame, i.e. already in plate coordinates."""
    sift = cv2.SIFT_create(nfeatures=6000)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    n = len(frames) if upto is None else upto + 1
    kp_prev, des_prev = sift.detectAndCompute(cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY), None)
    Hs = [np.eye(3)]
    for t in range(1, n):
        kp, des = sift.detectAndCompute(cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY), None)
        m = [a for a, b in bf.knnMatch(des, des_prev, k=2) if a.distance < 0.7 * b.distance]
        src = np.float32([kp[a.queryIdx].pt for a in m])
        dst = np.float32([kp_prev[a.trainIdx].pt for a in m])
        Hm, mk = cv2.findHomography(src, dst, cv2.RANSAC, 1.5)
        Hs.append(Hs[-1] @ Hm)
        kp_prev, des_prev = kp, des
    return [H / H[2, 2] for H in Hs]


BENCH_REGION = [(740, 140), (1110, 140), (1280, 300), (1280, 600), (1000, 600), (740, 430)]   # near bench + railing


def bench_homographies(frames, plate, people, min_inl=25, smooth=2):
    """Like ground_homographies, but fitted on the near bench and railing (seat/rail height, the same height as seated
    people), so seated people keep their true size and seat position when the generator's camera dollies in."""
    Hh, Ww = plate.shape[:2]
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)
    pg = cv2.cvtColor(plate.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    reg = np.zeros((Hh, Ww), np.uint8)
    cv2.fillPoly(reg, [np.array(BENCH_REGION, np.int32)], 255)
    kp_p, des_p = sift.detectAndCompute(pg, reg)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    Hs, used = [], []
    Hprev = np.eye(3)
    for i, f in enumerate(frames):
        fg = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        notp = (1 - cv2.dilate(people[i].astype(np.uint8), kd)) * 255
        kp_f, des_f = sift.detectAndCompute(fg, notp)
        H, u = Hprev, "prev"
        if des_f is not None and len(kp_f) > 30:
            m = [a for a, b in bf.knnMatch(des_f, des_p, k=2) if a.distance < 0.8 * b.distance]
            if len(m) >= min_inl:
                src = np.float32([kp_f[a.queryIdx].pt for a in m])
                dst = np.float32([kp_p[a.trainIdx].pt for a in m])
                Hm, mk = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
                if Hm is not None and mk.sum() >= min_inl:
                    H, u = Hm, f"bench{int(mk.sum())}"
        Hprev = H
        Hs.append(H / H[2, 2])
        used.append(u)
    C = np.float32([[0, 0], [Ww, 0], [Ww, Hh], [0, Hh]]).reshape(-1, 1, 2)
    P = np.stack([cv2.perspectiveTransform(C, H).reshape(4, 2) for H in Hs])
    Ps = P.copy()
    for i in range(len(P)):
        a, b = max(0, i - smooth), min(len(P), i + smooth + 1)
        Ps[i] = P[a:b].mean(0)
    Hs = np.stack([cv2.getPerspectiveTransform(C.reshape(4, 2), Ps[i].astype(np.float32)) for i in range(len(P))])
    return Hs, used
