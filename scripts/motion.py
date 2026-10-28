"""Quick numbers per take: motion in the action region vs background drift (vs plate) — helps rank takes.
  python scripts/motion.py work/beats/A01/r1/*.mkv
"""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from assemble import read_frames
plate = cv2.cvtColor(cv2.imread(os.path.join(__import__("sg_paths").ROOT, "work/plates/plate.png")), cv2.COLOR_BGR2RGB).astype(np.float32)
for p in sys.argv[1:]:
    if p.endswith("_mask.mkv"):
        continue
    f = read_frames(p).astype(np.float32)
    g = f.mean(-1)
    mot = np.abs(np.diff(g, axis=0))                  # per-pixel frame difference
    act = mot[:, 100:700, 450:1150].mean()            # action region (bench + walkway)
    bgd = np.abs(f[-1] - plate).mean(-1)              # last frame vs plate
    bg = np.median(bgd)                               # typical background drift
    print(f"{os.path.basename(p):24s} motion {act:5.2f}  bg_drift(median) {bg:5.2f}  last-vs-plate>25px {np.mean(bgd > 25) * 100:5.1f}%")
