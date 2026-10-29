"""Opening frame on the real view: move a seated person (cut from a walk-in take) along the bench to another seat.

  python scripts/move_seated.py --src work/start/real/P0a_g1_s3_f80.png --mask work/start/real/P0a_g1_s3_mask.png \
      --frac 0.30 --out work/start/real/A01_start.png

The bench front edge runs from the near-end armrest (911,366) to the far-end armrest (842,222) in 720p plate pixels;
the person moves --frac of that vector and is scaled by the measured perspective law h = 0.75 (foot_y + 36).
Pixels inside her silhouette hidden by the near armrest are filled from the surrounding coat before the move.
Her cast shadow is drawn with the calibrated sun model of compose.py.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import compose  # noqa: E402

from sg_paths import ROOT  # noqa: E402
NEAR, FAR = np.array([911.0, 366.0]), np.array([842.0, 222.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--frac", type=float, default=0.30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plate = cv2.cvtColor(cv2.imread(os.path.join(ROOT, "work", "plates", "plate.png")), cv2.COLOR_BGR2RGB)
    src = cv2.cvtColor(cv2.imread(a.src), cv2.COLOR_BGR2RGB)
    m = cv2.imread(a.mask, 0) > 127
    H, W = m.shape
    # silhouette with holes filled (the near armrest crosses her coat hem)
    # remove the woman from the plate first: the source frame is a generated take, so outside her silhouette we use
    # the plate itself; her old position shows the empty bench from the plate
    filled = np.zeros_like(m, np.uint8)
    cnts, _ = cv2.findContours(cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    holes = (filled > 0) & ~m
    srcf = cv2.inpaint(src, holes.astype(np.uint8), 5, cv2.INPAINT_TELEA) if holes.any() else src
    ys, xs = np.nonzero(filled)
    fy = float(np.percentile(ys, 99.5))
    fx = float(xs[ys > fy - 6].mean())
    d = a.frac * (FAR - NEAR)
    s = (fy + d[1] - compose.HORIZON) / (fy - compose.HORIZON)
    M = np.array([[s, 0, fx + d[0] - s * fx], [0, s, fy + d[1] - s * fy]], np.float32)
    mw = cv2.warpAffine(filled.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
    fw = cv2.warpAffine(srcf.astype(np.float32), M, (W, H), flags=cv2.INTER_AREA)
    # shadow of the moved silhouette, on sunlit platform only
    S, _ = compose.instance_shadow(mw > 0.5, H, W)
    lit = compose.lit_map(plate)
    S = cv2.GaussianBlur(S, (0, 0), 2.0) * lit
    gain = 1.0 - S[..., None] * 0.92 * (1.0 - compose.RATIO[None, None, :])
    # the bench's near armrest and front slats stay IN FRONT of her (occlusion), as in the plate
    occ = cv2.imread(os.path.join(ROOT, "work", "plates", "real", "bench_front_mask.png"), 0)
    if occ is None:
        print("WARNING: work/plates/real/bench_front_mask.png missing: the bench will not cover her (data bundle 'base')")
    occ = (occ > 127).astype(np.float32) if occ is not None else np.zeros((H, W), np.float32)
    alpha = cv2.GaussianBlur(mw, (0, 0), 0.8) * (1 - occ)
    alpha = alpha[..., None]
    out = alpha * fw + (1 - alpha) * plate.astype(np.float32) * gain
    out = out.round().clip(0, 255).astype(np.uint8)
    cv2.imwrite(a.out, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"moved by {d.round(1).tolist()} px, scale {s:.3f}, holes filled {int(holes.sum())} px -> {a.out}")


if __name__ == "__main__":
    main()
