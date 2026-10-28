"""Final per-beat composite: generated people over the fixed plate, with physically consistent cast shadows.

For each frame t of a kept take:
  stab_t   generated frame registered to the plate (ground-plane homography from lockjob)
  alpha_t  people matte: SAM3 masks warped to the plate, temporal median of 3, guided-filter edge refinement
  S_t      cast shadows: each person (SAM3 instance) is projected onto the platform along the sun direction
           measured on the plate (bench and lamp-post shadows: horizontal to the left, image length 1.5x height),
           thickened by body depth, penumbra growing with distance from the feet, plus a contact shadow at the feet
  out_t    = alpha_t * stab_t + (1 - alpha_t) * plate * gain_t,   gain_c = 1 - S_t * lit * (1 - ratio_c)
The generator's own shadows are dropped (they were weak or missing); `ratio` is the plate's measured shade/sun colour
ratio and `lit` keeps shadows from stacking on areas the plate already has in shade.

  python scripts/compose.py A05            # writes chain/A05.mkv + chain/A05_last.png (+ preview jpg)
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from assemble import read_frames  # noqa: E402

from sg_paths import ROOT  # noqa: E402
SUN_A, SUN_B = -0.93, -0.34                   # shadow offset per pixel of height (real GP02: walk-in shadows + post shadows)
RATIO = np.array([0.40, 0.44, 0.46], np.float32)   # plate shade / sun, R G B (real GP02 asphalt, histogram modes)
HORIZON = -36.0                               # ground-plane horizon (real GP02: woman's walk-in, h = 0.75 (y + 36))
_EDGE_Y = np.array([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720.0])
_EDGE_X = np.array([464, 435, 408, 383, 357, 332, 308, 283, 260, 237, 215, 193, 170, 149, 128, 107, 86, 66, 46.0])
EDGE = lambda y: np.interp(y, _EDGE_Y, _EDGE_X)     # platform edge (track side of the white coping), real GP02 view


def write_ffv1(frames, path, fps=16):
    import subprocess
    n, h, w, _ = frames.shape
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                    "-r", str(fps), "-i", "-", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", path],
                   input=np.ascontiguousarray(frames).tobytes(), check=True)


def guided(I, p, r=3, eps=1e-3):
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


def lit_map(plate):
    """1 where the platform is in sunlight, 0 inside the plate's own cast shadows (no double darkening).

    Shade is found by colour, not brightness: sunlight is warm and skylight in shadow is blue, so the plate's shadows
    have a clearly higher blue/red ratio than sunlit ground of the same material (measured 1.40 vs 0.92 on the asphalt,
    ~1.3 vs ~0.9 on the tactile paving). Thin dark joints, slab edges and paint lines keep the sun colour and stay lit.
    """
    p = cv2.GaussianBlur(plate.astype(np.float32), (0, 0), 2) + 1.0
    br = p[..., 2] / p[..., 0]
    Y = p.mean(-1)
    # asphalt shade: blue skylight (b/r > 1.12) and darker than sunlit asphalt; any surface: very dark (< 90),
    # which only shadows reach on this sunny plate (sunlit asphalt ~150, sunlit paving ~190)
    # real GP02 plate: sunlit asphalt ~105, shaded ~45 (b/r 0.93 vs 1.03); sunlit paving ~147, shaded ~60
    shade = ((Y < 72) | ((br > 1.0) & (Y < 80))).astype(np.uint8)
    shade = cv2.morphologyEx(shade, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    shade = cv2.morphologyEx(shade, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(shade)
    big = np.zeros(n, bool)
    big[1:] = st[1:, cv2.CC_STAT_AREA] > 800
    lit = 1.0 - cv2.GaussianBlur(big[lab].astype(np.float32), (0, 0), 2.0)
    H, W = lit.shape
    yy, xx = np.mgrid[0:H, 0:W]
    platform = np.clip((xx - EDGE(yy)) / 3.0, 0, 1)          # shadows only on the platform surface
    return (lit * platform).astype(np.float32)


def instance_shadow(mask, H, W):
    """Shadow coverage (0..1, float32) of one person on the ground plane, and the foot point."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 400:
        return None, None
    if ys.max() >= H - 2 and (xs[ys >= H - 2].size > 0):
        pass                                                   # feet at the frame edge: shadow still cast from there
    fy = float(np.percentile(ys, 99.5))
    bottom = ys > fy - 0.04 * (fy - ys.min() + 1)
    fx = float(xs[bottom].mean())
    half_k = 0.85 * (fy - HORIZON) / 1012.0 / 2.0           # body depth / width, depth scale / lateral scale
    S = np.zeros((H, W), np.float32)
    order = np.argsort(ys, kind="stable")
    ys_s, xs_s = ys[order], xs[order]
    rows, starts, counts = np.unique(ys_s, return_index=True, return_counts=True)
    for y, s0, c in zip(rows, starts, counts):
        xr = xs_s[s0:s0 + c]
        v = max(0.0, fy - y)
        x0, x1 = xr.min() + SUN_A * v, xr.max() + SUN_A * v
        yc = fy + SUN_B * v
        t = max(1.5, half_k * c)
        cv2.rectangle(S, (int(round(x0)), int(round(yc - t))), (int(round(x1)), int(round(yc + t))), 1.0, -1)
    return S, (fx, fy)


def shadows_for_frame(inst_masks, H, W):
    """Union of all people's cast shadows with distance-dependent penumbra, plus contact shadows."""
    Sall = np.zeros((H, W), np.float32)
    ao = np.zeros((H, W), np.float32)
    xx = np.arange(W, dtype=np.float32)[None, :]
    for m in inst_masks:
        S, foot = instance_shadow(m, H, W)
        if S is None:
            continue
        fx, fy = foot
        d = np.clip(fx - xx, 0, None)                        # distance from the feet along the shadow
        s1, s2, s3 = (cv2.GaussianBlur(S, (0, 0), s) for s in (1.0, 3.0, 7.0))
        w2 = np.clip(d / 300.0, 0, 1)
        w3 = np.clip((d - 300.0) / 500.0, 0, 1)
        Ssoft = (1 - w2) * s1 + w2 * ((1 - w3) * s2 + w3 * s3)
        Sall = np.maximum(Sall, np.broadcast_to(Ssoft, Sall.shape))
        # contact shadow: soft dark ellipse under the feet
        ys, xs = np.nonzero(m)
        width = max(10.0, float(np.ptp(xs[ys > fy - 0.05 * (fy - ys.min() + 1)])))
        e = np.zeros((H, W), np.float32)
        cv2.ellipse(e, (int(fx), int(fy)), (int(0.75 * width), max(3, int(0.12 * width))), 0, 0, 360, 1.0, -1)
        ao = np.maximum(ao, cv2.GaussianBlur(e, (0, 0), 4.0))
    return Sall, ao


def drop_small_blobs(m, rel=0.06, min_abs=600):
    """Remove detached mask fragments (generator smears SAM3 calls 'person') much smaller than the people."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 2:
        return m
    big = st[1:, cv2.CC_STAT_AREA].max()
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= max(min_abs, rel * big)
    return keep[lab]


def person_alpha(stab, masks):
    """Temporal median-of-3 on the binary masks, then guided-filter refinement against the frame."""
    n = len(masks)
    masks = np.stack([drop_small_blobs(m) for m in masks])
    med = masks.copy()
    for i in range(n):
        a, b = masks[max(0, i - 1)], masks[min(n - 1, i + 1)]
        med[i] = (masks[i].astype(np.uint8) + a + b) >= 2
    out = np.zeros(masks.shape, np.float32)
    k = np.ones((3, 3), np.uint8)
    for i in range(n):
        # tight edge: 1 px dilation, 1 px Gaussian feather (wide guided-filter edges let the generator's background
        # bleed through where the person and the bench are both dark)
        p = cv2.dilate(med[i].astype(np.uint8), k).astype(np.float32)
        out[i] = np.clip(cv2.GaussianBlur(p, (0, 0), 1.0), 0, 1)
    return out


def compose_beat(beat, write=True):
    c = json.load(open(os.path.join(ROOT, "chain", "chain.json")))
    rec = c["beats"][beat]
    wd = os.path.join(ROOT, "work", "lock", beat)
    raw = read_frames(rec["take"])
    Hs = np.load(os.path.join(wd, "homographies.npy"))
    union = np.load(os.path.join(wd, "people.npz"))["masks"]
    inst = np.load(os.path.join(wd, "instances.npz"))["labels"]
    plate = cv2.cvtColor(cv2.imread(c["plate"]), cv2.COLOR_BGR2RGB)
    H, W = plate.shape[:2]
    n = len(raw)
    warp = lambda img, Hm, fl: cv2.warpPerspective(img, Hm, (W, H), flags=fl)
    stab = np.stack([warp(raw[i], Hs[i], cv2.INTER_LINEAR) for i in range(n)])
    masks = np.stack([warp(union[i].astype(np.uint8), Hs[i], cv2.INTER_NEAREST) > 0 for i in range(n)])
    alpha = person_alpha(stab, masks)
    lit = lit_map(plate)
    pf = plate.astype(np.float32)
    out = np.empty_like(stab)
    shadow_cov = []
    s_all = np.zeros((n, H, W), np.uint8)
    for i in range(n):
        lab = warp(inst[i], Hs[i], cv2.INTER_NEAREST)
        ims = [lab == k for k in range(1, int(lab.max()) + 1) if (lab == k).sum() > 400]
        S, ao = shadows_for_frame(ims, H, W)
        s = np.clip(S * 0.92 + ao * 0.35, 0, 1) * lit
        s_all[i] = (s * 255 + 0.5).astype(np.uint8)
        gain = 1.0 - s[..., None] * (1.0 - RATIO[None, None, :])
        a = alpha[i][..., None]
        out[i] = (a * stab[i] + (1 - a) * pf * gain).round().clip(0, 255).astype(np.uint8)
        shadow_cov.append(float(S.mean()))
    if write:
        clip = os.path.join(ROOT, "chain", f"{beat}.mkv")
        write_ffv1(out, clip)
        np.savez_compressed(os.path.join(wd, "matte.npz"), alpha=(alpha * 255 + 0.5).astype(np.uint8), shadow=s_all)
        cv2.imwrite(os.path.join(ROOT, "chain", f"{beat}_last.png"), cv2.cvtColor(out[-1], cv2.COLOR_RGB2BGR))
        rows = [np.concatenate([cv2.resize(out[i], (640, 360)),
                                cv2.resize((np.repeat(alpha[i][..., None], 3, 2) * 255).astype(np.uint8), (640, 360))], 1)
                for i in (0, 40, 80)]
        cv2.imwrite(os.path.join(wd, "compose_preview.jpg"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
        c = json.load(open(os.path.join(ROOT, "chain", "chain.json")))
        c["beats"][beat].update(clip=clip, lock="compose v3: sam3 matte + geometric shadows",
                                shadow_cover_mean=round(float(np.mean(shadow_cov)), 4))
        json.dump(c, open(os.path.join(ROOT, "chain", "chain.json") + ".tmp", "w"), indent=1)
        os.replace(os.path.join(ROOT, "chain", "chain.json") + ".tmp", os.path.join(ROOT, "chain", "chain.json"))
    return out, alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("beats", nargs="+")
    a = ap.parse_args()
    for b in a.beats:
        compose_beat(b)
        print("composed", b, flush=True)


if __name__ == "__main__":
    main()
