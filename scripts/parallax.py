"""Parallax check for a take: after ground registration, how well do raised structures (benches, railing, posts) line
up with the plate? Edge-distance median/p90 in px over the bench+railing band, people excluded by a coarse mask.
  python scripts/parallax.py work/beats/A07/g1/A07_g1_s3.mkv [more...]
"""
import os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import bglock
from assemble import read_frames
pl = cv2.cvtColor(cv2.imread(os.path.join(__import__("sg_paths").ROOT, "work/plates/plate.png")), cv2.COLOR_BGR2RGB)
e1 = cv2.Canny(cv2.cvtColor(pl, cv2.COLOR_RGB2GRAY), 60, 150)
dt = cv2.distanceTransform(255 - e1, cv2.DIST_L2, 3)
band = np.zeros(pl.shape[:2], bool); cv2.fillPoly(band.view(np.uint8), [np.array([(640, 0), (1280, 0), (1280, 720), (1000, 720), (760, 160)], np.int32)], 1)
band = band.view(np.uint8).astype(bool)
for p in sys.argv[1:]:
    R = read_frames(p)
    idx = [20, 40, 60, 80]
    sel = R[idx]
    Hs, _ = bglock.ground_homographies(sel, pl, np.zeros(sel.shape[:3], bool), smooth=0)
    out = []
    for f, H in zip(sel, Hs):
        w = cv2.warpPerspective(f, H, (1280, 720))
        d = np.abs(w.astype(np.int16) - pl.astype(np.int16)).max(-1) > 60          # people & big changes
        d = cv2.dilate(d.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
        e2 = cv2.Canny(cv2.cvtColor(w, cv2.COLOR_RGB2GRAY), 60, 150) > 0
        sel_e = e2 & band & ~d & (w.sum(-1) > 0)
        v = dt[sel_e]
        out.append(f"{np.median(v):.1f}/{np.percentile(v, 90):.1f}")
    print(os.path.basename(p)[:-4], "edge err median/p90 at f20,40,60,80:", "  ".join(out))
