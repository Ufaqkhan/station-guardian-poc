"""Registered review grid for a beat's takes (fixed view, natural colours). Waits for --n takes.
  python scripts/grid.py A11 g2 --idx 16 32 48 64 80 --box 760 100 1120 430
Prints camera scale per sampled frame; writes work/qa/<beat>_<tag>.jpg
"""
import argparse, glob, os, sys, time
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import bglock
from assemble import read_frames
from sg_paths import ROOT as R0  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("beat"); ap.add_argument("tag")
ap.add_argument("--n", type=int, default=6); ap.add_argument("--idx", type=int, nargs="+", default=[16, 32, 48, 64, 80])
ap.add_argument("--box", type=int, nargs=4, default=[760, 100, 1120, 430]); ap.add_argument("--w", type=int, default=240)
ap.add_argument("--raw", action="store_true", help="no registration")
a = ap.parse_args()
d = os.path.join(R0, "work", "beats", a.beat, a.tag)
while len(glob.glob(os.path.join(d, "*_sheet.jpg"))) < a.n:
    time.sleep(10)
time.sleep(5)
pl = cv2.cvtColor(cv2.imread(os.path.join(R0, "work/plates/plate.png")), cv2.COLOR_BGR2RGB)
x0, y0, x1, y1 = a.box; th = int(a.w * (y1 - y0) / (x1 - x0))
rows = []
for mkv in sorted(glob.glob(os.path.join(d, "*.mkv")), key=lambda p: int(p.rsplit("_s", 1)[1][:-4])):
    R = read_frames(mkv); sel = R[[min(i, len(R) - 1) for i in a.idx]]
    Hs = [np.eye(3)] * len(sel) if a.raw else bglock.ground_homographies(sel, pl, np.zeros(sel.shape[:3], bool), smooth=0)[0]
    name = mkv.rsplit("_", 1)[1][:-4]; t = []
    for i, f, H in zip(a.idx, sel, Hs):
        w = f if a.raw else cv2.warpPerspective(f, H, (1280, 720), flags=cv2.INTER_AREA)
        x = cv2.resize(np.ascontiguousarray(w[y0:y1, x0:x1]), (a.w, th))
        cv2.putText(x, f"{name} f{i} {H[0, 0]:.2f}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1); t.append(x)
    rows.append(np.concatenate(t, 1))
half = (len(rows) + 1) // 2
L = rows[:half]; Rr = rows[half:] + [np.zeros_like(rows[0])] * (half - len(rows[half:]))
out = os.path.join(R0, "work", "qa", f"{a.beat}_{a.tag}.jpg")
cv2.imwrite(out, cv2.cvtColor(np.concatenate([np.concatenate(L, 0), np.concatenate(Rr, 0)], 1), cv2.COLOR_RGB2BGR))
print(out)
