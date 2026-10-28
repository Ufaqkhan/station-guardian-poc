"""Pinned end frame = empty plate + BOTH people (SAM3 'person' masks, grown to touching changed pixels) from one
registered take frame. Avoids ghosts from the previous beat's frame.
  python scripts/end_from_take.py --src work/keyframes/real/A14/s6_f28_reg.png --out work/keyframes/real/A14/A14_end.png
"""
import argparse, json, os, sys, time
import cv2, numpy as np
from sg_paths import ROOT as R0, SAM3_PY as SAM3, SCRIPTS  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
stem = os.path.splitext(a.out)[0]
masks = []
for who in ("man in a green jacket", "woman in a red coat"):
    mp = stem + "_" + ("man" if who.startswith("man") else "woman") + "_mask.png"
    if not os.path.exists(mp):
        q = os.path.join(R0, "work", "queue"); n = f"efm_{os.path.basename(mp)[:-4]}_{int(time.time())}"
        cmd = f"{SAM3} {SCRIPTS}/sam3_instance.py --image {os.path.abspath(a.src)} --prompt '{who}' --out {mp}"
        json.dump(dict(name=n, cmd=cmd, priority=-1), open(os.path.join(q, n + ".json.tmp"), "w"))
        os.rename(os.path.join(q, n + ".json.tmp"), os.path.join(q, n + ".json"))
        while not os.path.exists(mp):
            time.sleep(2)
        time.sleep(0.5)
    masks.append(cv2.imread(mp, 0) > 127)
pl = cv2.imread(os.path.join(R0, "work/plates/plate.png")).astype(np.float32)
src = cv2.imread(a.src).astype(np.float32)
m = masks[0] | masks[1]
d = np.abs(src - pl).max(-1) > 40
m = m | (d & (cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0))
m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
al = cv2.GaussianBlur(cv2.dilate(m, np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 1.0)[..., None]
cv2.imwrite(a.out, (pl * (1 - al) + src * al).round().clip(0, 255).astype(np.uint8))
print("wrote", a.out)
