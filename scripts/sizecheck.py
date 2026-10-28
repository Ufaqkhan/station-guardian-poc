"""Apparent size stability of the seated woman after ground registration: area of her white beanie (very bright,
low-saturation blob above the bench) per frame, relative to frame 0. A real seated person keeps a constant size.
  python scripts/sizecheck.py take.mkv [...]
"""
import os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import bglock
from assemble import read_frames
pl = cv2.cvtColor(cv2.imread(os.path.join(__import__("sg_paths").ROOT, "work/plates/plate.png")), cv2.COLOR_BGR2RGB)
for p in sys.argv[1:]:
    R = read_frames(p); idx = list(range(0, 81, 10)); sel = R[idx]
    Hs, _ = bglock.ground_homographies(sel, pl, np.zeros(sel.shape[:3], bool), smooth=0)
    areas = []
    for f, H in zip(sel, Hs):
        w = cv2.warpPerspective(f, H, (1280, 720)).astype(np.int16)
        roi = w[60:330, 760:1000]
        bright = (roi.min(-1) > 175) & ((roi.max(-1) - roi.min(-1)) < 40)
        n, lab, st, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8))
        areas.append(st[1:, 4].max() if n > 1 else 0)
    a = np.array(areas, float); r = a / max(a[0], 1)
    print(os.path.basename(p)[:-4], "beanie area rel f0:", " ".join(f"{x:.2f}" for x in r))
