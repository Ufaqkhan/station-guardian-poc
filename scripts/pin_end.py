"""Build a pinned end frame in the fixed GP02 view: take the previous beat's last frame, replace one person with that
person's pose from a registered take frame.

  python scripts/pin_end.py --base chain/A03_last_cc.png --src work/keyframes/real/A04s3_f80_reg.png \
      --who "man in a green jacket" --out work/keyframes/real/A04_end.png

1. SAM3 masks for `--who` on both images (queued on a worker GPU).
2. The person's old silhouette (and a margin for his old shadow) is replaced by the plate in the base.
3. The person from the source (registered to the plate) is pasted with a 1 px feathered matte; his shadow is left to
   compose.py, which re-renders all shadows geometrically in the final composite.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from sg_paths import ROOT, SAM3_PY as SAM3, SCRIPTS  # noqa: E402


def sam3_mask(img_path, prompt, out_path):
    q = os.path.join(ROOT, "work", "queue")
    n = f"pin_{os.path.basename(out_path)[:-4]}_{int(time.time() * 1000) % 10**8}"
    cmd = f"{SAM3} {SCRIPTS}/sam3_instance.py --image {img_path} --prompt '{prompt}' --out {out_path}"
    p = os.path.join(q, n + ".json")
    json.dump(dict(name=n, cmd=cmd, priority=-1), open(p + ".tmp", "w"))
    os.rename(p + ".tmp", p)
    while not os.path.exists(out_path):
        if any(n in f for f in os.listdir(os.path.join(q, "failed"))):
            raise RuntimeError("sam3 failed for " + img_path)
        time.sleep(2)
    time.sleep(0.5)
    return cv2.imread(out_path, 0) > 127


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--who", default="man in a green jacket")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep_base_mask", default=None, help="mask (png) of the OTHER person in the base to protect")
    ap.add_argument("--base_beat", default=None, help="beat whose last frame is --base: its rendered shadows are erased")
    a = ap.parse_args()
    base = cv2.imread(a.base).astype(np.float32)
    src = cv2.imread(a.src).astype(np.float32)
    plate = cv2.imread(os.path.join(ROOT, "work", "plates", "plate.png")).astype(np.float32)
    stem = os.path.splitext(a.out)[0]
    m_old = sam3_mask(os.path.abspath(a.base), a.who, stem + "_oldmask.png")
    m_new = sam3_mask(os.path.abspath(a.src), a.who, stem + "_newmask.png")
    # remove the old pose (+ margin and his old ground shadow region left of his feet) unless it is the other person
    old = cv2.dilate(m_old.astype(np.uint8), np.ones((9, 9), np.uint8))
    ys, xs = np.nonzero(m_old)
    if len(ys):
        fy, fx = ys.max(), int(xs[ys > ys.max() - 6].mean())
        cv2.ellipse(old, (fx - 60, fy - 15), (110, 40), -20, 0, 360, 1, -1)
    if a.keep_base_mask:
        other = cv2.imread(a.keep_base_mask, 0) > 127
        old[cv2.dilate(other.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 0
    o = cv2.GaussianBlur(old.astype(np.float32), (0, 0), 1.5)[..., None]
    out = base * (1 - o) + plate * o
    if a.base_beat:
        # undo the compositor's cast shadows of the base frame (compose.py re-renders shadows for the new beat)
        mt = np.load(os.path.join(ROOT, "work", "lock", a.base_beat, "matte.npz"))
        sh = mt["shadow"][-1].astype(np.float32) / 255.0
        ppl = mt["alpha"][-1].astype(np.float32) / 255.0
        w = np.clip(cv2.GaussianBlur((sh > 0.02).astype(np.float32), (0, 0), 1.5) * (1 - ppl), 0, 1)[..., None]
        out = out * (1 - w) + plate * w
    al = cv2.GaussianBlur(cv2.dilate(m_new.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 1.0)
    out = out * (1 - al[..., None]) + src * al[..., None]
    cv2.imwrite(a.out, out.round().clip(0, 255).astype(np.uint8))
    ys, xs = np.nonzero(m_new)
    print(f"wrote {a.out}; new foot ({xs[ys > ys.max() - 6].mean():.0f},{ys.max()}), top {ys.min()}")


if __name__ == "__main__":
    main()
