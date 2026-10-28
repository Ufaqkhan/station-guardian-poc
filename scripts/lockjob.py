"""Commit one take to the chain with a background lock built from SAM3 person masks.

  python scripts/lockjob.py A02 work/beats/A02/r1/A02_r1_s3.mkv [--no_queue_next]

1. stabilise every frame onto the plate (SIFT homography + ECC), CPU
2. SAM3 masks for "person" and "handbag" on the stabilised frames: queued as a command request on the
   Wan worker's GPU (SAM3 needs CUDA), this script waits for the result
3. cast shadows: pixels near a person that are clearly darker than the plate
4. composite people + shadows onto the plate, feathered, temporally smoothed
5. write chain/<beat>.mkv, chain/<beat>_last.png, a mask preview, and record it in chain/chain.json
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import bglock  # noqa: E402
import chain as ch  # noqa: E402
from assemble import read_frames  # noqa: E402

ROOT = ch.ROOT
from sg_paths import SAM3_PY, SCRIPTS  # noqa: E402,F401


def run_sam3(beat, frames_npy, out_npz, prompts=("person", "handbag")):
    q = os.path.join(ROOT, "work", "queue")
    name = f"sam3_{beat}_{int(time.time())}"
    cmd = (f"{SAM3_PY} {SCRIPTS}/sam3_people.py --frames {frames_npy} --out {out_npz} --device cuda "
           f"--prompts {' '.join(repr(p) for p in prompts)}")
    req = dict(name=name, cmd=cmd, priority=-1, submitted=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = os.path.join(q, f".{name}.tmp")
    json.dump(req, open(tmp, "w"))
    os.rename(tmp, os.path.join(q, name + ".json"))
    print("queued", name, flush=True)
    while True:
        done = [p for p in os.listdir(os.path.join(q, "done")) if p.endswith(f"__{name}.json")]
        fail = [p for p in os.listdir(os.path.join(q, "failed")) if p.endswith(f"__{name}.json")]
        if done:
            return json.load(open(os.path.join(q, "done", done[0])))
        if fail:
            raise RuntimeError(json.load(open(os.path.join(q, "failed", fail[0]))).get("error", "sam3 failed"))
        time.sleep(3)


def build_alpha(stab, plate, people, shadow_ratio=0.80, search=90, feather=2.5, chroma_tol=9.0):
    """Returns (alpha_people, gain, shadow_px).

    alpha_people: soft matte of the people (SAM3 masks, dropout-filled, temporally widened) -> generated pixels.
    gain: per-pixel brightness factor applied to the PLATE for cast shadows. A pixel counts as shadow only if it is
    darker than the plate by the given ratio AND keeps the plate's colour (Lab a/b within chroma_tol), near a person.
    So shadows always show the real ground texture and colour; coloured junk the generator leaves on the ground
    (blue blotches, a redrawn armrest) is rejected and the plate shows instead.
    """
    n, H, W = people.shape
    P = people.copy()
    for i in range(n):
        if not P[i].any():
            nb = [P[j] for j in (i - 1, i + 1) if 0 <= j < n and people[j].any()]
            if nb:
                P[i] = np.logical_or.reduce(nb)
    T = P.copy()
    T[1:] |= P[:-1]
    T[:-1] |= P[1:]
    k4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ks = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * search + 1, 2 * search + 1))
    kt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    plab = cv2.GaussianBlur(cv2.cvtColor(plate, cv2.COLOR_RGB2LAB).astype(np.float32), (0, 0), 2)
    pg = cv2.GaussianBlur(cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 2) + 1.0
    alpha = np.zeros((n, H, W), np.float32)
    ratio = np.ones((n, H, W), np.float32)
    smask = np.zeros((n, H, W), np.float32)
    shadow_px = []
    for i in range(n):
        pm = cv2.dilate(T[i].astype(np.uint8), k4)
        g = cv2.GaussianBlur(cv2.cvtColor(stab[i], cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 2) + 1.0
        lab = cv2.GaussianBlur(cv2.cvtColor(stab[i], cv2.COLOR_RGB2LAB).astype(np.float32), (0, 0), 2)
        r = g / pg
        chroma_ok = np.sqrt(((lab[..., 1:] - plab[..., 1:]) ** 2).sum(-1)) < chroma_tol
        dark = ((r < shadow_ratio) & chroma_ok).astype(np.uint8)
        region = cv2.dilate(pm, ks) & (1 - pm)
        s = dark & region
        s = cv2.morphologyEx(s, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        s = cv2.morphologyEx(s, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        near = cv2.dilate(pm, kt)
        nlab, cc = cv2.connectedComponents(s)
        keep = np.zeros_like(s)
        for c in range(1, nlab):
            comp = cc == c
            if (comp & (near > 0)).any() and comp.sum() > 80:
                keep[comp] = 1
        shadow_px.append(int(keep.sum()))
        alpha[i] = pm.astype(np.float32)
        smask[i] = keep.astype(np.float32)
        ratio[i] = np.clip(r, 0.3, 1.0)
    def tsmooth(x):
        y = x.copy()
        y[1:-1] = (x[:-2] + 2 * x[1:-1] + x[2:]) / 4
        return y
    a2 = np.maximum(tsmooth(alpha), alpha * 0.999)
    alpha = np.stack([cv2.GaussianBlur(a, (0, 0), feather) for a in a2]).clip(0, 1)
    sm = np.stack([cv2.GaussianBlur(a, (0, 0), 3.0) for a in tsmooth(smask)]).clip(0, 1)
    rs = np.stack([cv2.GaussianBlur(a, (0, 0), 2.0) for a in tsmooth(ratio)])
    gain = 1.0 - sm * (1.0 - rs)
    return alpha, gain, shadow_px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("beat")
    ap.add_argument("take")
    ap.add_argument("--shadow_ratio", type=float, default=0.80)
    ap.add_argument("--next", default=None, help="queue this beat from the new last frame when done")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--extra", default="")
    ap.add_argument("--priority", default="1")
    ap.add_argument("--reuse_masks", action="store_true", help="reuse work/lock/<beat>/people.npz for the same take")
    ap.add_argument("--reg", choices=["ground", "bench"], default=os.environ.get("REG", "ground"),
                    help="registration surface: ground (walking beats) or bench (seated beats)")
    args = ap.parse_args()
    c = ch.load()
    t0 = time.time()
    fr = read_frames(args.take)
    plate = cv2.cvtColor(cv2.imread(c["plate"]), cv2.COLOR_BGR2RGB)
    wd = os.path.join(ROOT, "work", "lock", args.beat)
    os.makedirs(wd, exist_ok=True)
    # 1) people masks on the RAW frames (GPU, SAM3)
    npy = os.path.join(wd, "raw.npy")
    np.save(npy, fr)
    npz = os.path.join(wd, "people.npz")
    if args.reuse_masks and os.path.exists(npz) and json.load(open(os.path.join(ch.CH, "chain.json")))["beats"].get(
            args.beat, {}).get("take") == os.path.abspath(args.take):
        print("reusing SAM3 masks", npz, flush=True)
    else:
        r = run_sam3(args.beat, npy, npz)
        print("sam3:", r.get("stdout", "").strip()[-200:], flush=True)
    people_raw = np.load(npz)["masks"]
    # 2) per-frame homography fitted on the context ring around the people, smoothed; warp frames + masks
    Hs, used = (bglock.bench_homographies if args.reg == "bench" else bglock.ground_homographies)(fr, plate, people_raw)
    print("registration:", args.reg, flush=True)
    stab, people = bglock.warp_all(fr, people_raw, Hs, plate.shape)
    inl = [int(''.join(ch for ch in u if ch.isdigit()) or 0) for u in used]
    print(f"aligned in {time.time() - t0:.0f}s; fits {sum(not u.startswith('prev') for u in used)}/{len(used)}, "
          f"global {sum(u.startswith('global') for u in used)}", flush=True)
    alpha, gain, shadow_px = build_alpha(stab, plate, people, shadow_ratio=args.shadow_ratio)
    a = alpha[..., None]
    out = (a * stab.astype(np.float32) + (1 - a) * plate.astype(np.float32) * gain[..., None]
           ).round().clip(0, 255).astype(np.uint8)
    os.makedirs(ch.CH, exist_ok=True)
    clip = os.path.join(ch.CH, f"{args.beat}.mkv")
    bglock.write_ffv1(out, clip)
    cv2.imwrite(os.path.join(ch.CH, f"{args.beat}_last.png"), cv2.cvtColor(out[-1], cv2.COLOR_RGB2BGR))
    # preview: frames 0/40/80 composite + matte
    rows = []
    for i in (0, 40, 80):
        m = np.stack([alpha[i], 1 - gain[i], np.zeros_like(alpha[i])], -1)
        m = (m.clip(0, 1) * 255).astype(np.uint8)
        rows.append(np.concatenate([cv2.resize(out[i], (640, 360)), cv2.resize(m, (640, 360))], 1))
    cv2.imwrite(os.path.join(wd, "lock_preview.jpg"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    rec = dict(take=os.path.abspath(args.take), clip=clip, lock="sam3+shadow",
               cover_mean=round(float(alpha.mean()), 3), people_per_frame=np.load(npz)["counts"].tolist(),
               shadow_px_mean=int(np.mean(shadow_px)), homography_inliers_min=int(min(inl) if inl else 0))
    meta = args.take[:-4] + ".json"
    if os.path.exists(meta):
        m = json.load(open(meta))
        rec.update(prompt=m.get("prompt"), seed=m.get("seed"), start=m.get("image"))
    c = ch.load()
    c["beats"][args.beat] = rec
    ch.save(c)
    os.remove(npy)
    np.save(os.path.join(wd, "homographies.npy"), Hs)
    print(f"locked {args.beat} in {time.time() - t0:.0f}s; cover {rec['cover_mean']}; "
          f"people/frame {sorted(set(rec['people_per_frame']))}; preview {wd}/lock_preview.jpg", flush=True)
    if args.next:
        import subprocess
        # colour drift reset: garment colours of the last frame back towards the reference before the next beat
        import colorfix
        colorfix.measure(args.beat)
        r2 = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "colorfix.py"), "seed", args.beat],
                            capture_output=True, text=True)
        print(r2.stdout[-600:], flush=True)
        start = os.path.join(ch.CH, f"{args.beat}_last_cc.png")
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "submit.py"), "--beat", args.next, "--start",
               start, "--priority", args.priority, "--seeds"] + [str(x) for x in args.seeds]
        if args.extra:
            cmd += ["--extra", args.extra]
        print(subprocess.run(cmd, capture_output=True, text=True).stdout, flush=True)


if __name__ == "__main__":
    main()
