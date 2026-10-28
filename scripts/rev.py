"""Review all takes of a beat as they will appear in the final fixed view: wait until N takes exist, register each
sampled frame to the plate (ground homography), crop, brighten for inspection, one row per take; print parallax.
  python scripts/rev.py A11 g1 --n 6 --box 790 110 1070 400 --idx 16 32 48 64 80
"""
import argparse, glob, os, sys, time
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import bglock
from assemble import read_frames

from sg_paths import ROOT as R0  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("beat"); ap.add_argument("tag")
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--box", type=int, nargs=4, default=[790, 110, 1070, 400])
ap.add_argument("--idx", type=int, nargs="+", default=[16, 32, 48, 64, 80])
ap.add_argument("--w", type=int, default=230)
ap.add_argument("--bright", type=float, default=1.2)
a = ap.parse_args()
d = os.path.join(R0, "work", "beats", a.beat, a.tag)
while len(glob.glob(os.path.join(d, "*_sheet.jpg"))) < a.n:
    if glob.glob(os.path.join(R0, "work", "queue", "failed", f"*{a.beat}_{a.tag}_*")):
        print("FAILED takes present"); break
    time.sleep(10)
pl = cv2.cvtColor(cv2.imread(os.path.join(R0, "work/plates/plate.png")), cv2.COLOR_BGR2RGB)
e1 = cv2.Canny(cv2.cvtColor(pl, cv2.COLOR_RGB2GRAY), 60, 150)
dt = cv2.distanceTransform(255 - e1, cv2.DIST_L2, 3)
band = np.zeros(pl.shape[:2], np.uint8)
cv2.fillPoly(band, [np.array([(640, 0), (1280, 0), (1280, 720), (1000, 720), (760, 160)], np.int32)], 1)
x0, y0, x1, y1 = a.box
th = int(a.w * (y1 - y0) / (x1 - x0))
rows = []
for mkv in sorted(glob.glob(os.path.join(d, "*.mkv")), key=lambda p: int(p.rsplit("_s", 1)[1][:-4])):
    R = read_frames(mkv); sel = R[[min(i, len(R) - 1) for i in a.idx]]
    Hs, _ = bglock.ground_homographies(sel, pl, np.zeros(sel.shape[:3], bool), smooth=0)
    tiles, errs = [], []
    name = os.path.basename(mkv)[:-4].rsplit("_", 1)[1]
    for i, f, H in zip(a.idx, sel, Hs):
        w = cv2.warpPerspective(f, H, (1280, 720), flags=cv2.INTER_CUBIC)
        dd = cv2.dilate((np.abs(w.astype(np.int16) - pl.astype(np.int16)).max(-1) > 60).astype(np.uint8), np.ones((25, 25), np.uint8))
        e2 = (cv2.Canny(cv2.cvtColor(w, cv2.COLOR_RGB2GRAY), 60, 150) > 0) & (band > 0) & (dd == 0) & (w.sum(-1) > 0)
        errs.append(np.percentile(dt[e2], 90) if e2.any() else 99)
        t = cv2.convertScaleAbs(np.ascontiguousarray(w[y0:y1, x0:x1]), alpha=a.bright, beta=10)
        t = cv2.resize(t, (a.w, th)); cv2.putText(t, f"{name} f{i}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
        tiles.append(t)
    rows.append(np.concatenate(tiles, 1))
    print(f"{name}: parallax p90 per frame", " ".join(f"{e:.1f}" for e in errs))
out = os.path.join(d, "rev.jpg")
cv2.imwrite(out, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(out)
