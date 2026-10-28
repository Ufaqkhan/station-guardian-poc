"""One grid per beat: a row per take, 5 moments (f0..f80), cropped to the action area.
  python scripts/review.py A03 r1 [--box x0 y0 x1 y1] [--scale 0.42]
"""
import argparse, glob, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from assemble import read_frames
from sg_paths import ROOT
os.chdir(ROOT)  # paths below are relative to the data folder
ap = argparse.ArgumentParser(); ap.add_argument("beat"); ap.add_argument("tag")
ap.add_argument("--box", type=int, nargs=4, default=[280, 0, 1180, 720]); ap.add_argument("--scale", type=float, default=0.42)
ap.add_argument("--idx", type=int, nargs="+", default=[0, 20, 40, 60, 80])
ap.add_argument("--reg", action="store_true", help="register each frame to the plate first (final fixed view)")
a = ap.parse_args()
x0, y0, x1, y1 = a.box
rows = []
for mkv in sorted(glob.glob(f"work/beats/{a.beat}/{a.tag}/*.mkv")):
    f = read_frames(mkv); tiles = []
    if a.reg:
        import bglock
        _pl = cv2.cvtColor(cv2.imread("work/plates/plate.png"), cv2.COLOR_BGR2RGB)
        sel = f[[min(i, len(f) - 1) for i in a.idx]]
        _Hs, _ = bglock.ground_homographies(sel, _pl, np.zeros(sel.shape[:3], bool), smooth=0)
        regd = {i: cv2.warpPerspective(fr, H, (_pl.shape[1], _pl.shape[0]), flags=cv2.INTER_AREA) for i, fr, H in zip(a.idx, sel, _Hs)}
    for i in a.idx:
        src = regd[i] if a.reg else f[min(i, len(f) - 1)]
        t = cv2.resize(np.ascontiguousarray(src[y0:y1, x0:x1]), None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
        cv2.putText(t, f"{os.path.basename(mkv)[:-4].split('_')[-1]} f{i}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(t)
    rows.append(np.concatenate(tiles, 1))
out = f"work/beats/{a.beat}/{a.tag}/review.jpg"
cv2.imwrite(out, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88]); print(out)
# camera movement per take (ground-plane homography at f40, f80 relative to the plate)
import bglock
plate = cv2.cvtColor(cv2.imread("work/plates/plate.png"), cv2.COLOR_BGR2RGB)
for mkv in sorted(glob.glob(f"work/beats/{a.beat}/{a.tag}/*.mkv")):
    R = read_frames(mkv)[[0, 40, 80]]
    Hs, u = bglock.ground_homographies(R, plate, np.zeros(R.shape[:3], bool), smooth=0)
    print(os.path.basename(mkv)[:-4], " ".join("f%d s=%.3f tx=%+.0f ty=%+.0f" % (f, h[0, 0], h[0, 2], h[1, 2]) for f, h in zip((40, 80), Hs[1:])))
