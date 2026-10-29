"""Save one frame of a take as PNG, optionally registered to the plate, optionally with a SAM3 mask.

Used to build the opening frame and the end frames of pinned beats:

  # opening frame source: frame 80 of the walk-in take + mask of the seated woman (as run for the PoC)
  python scripts/take_frame.py work/beats/P0a/g1/P0a_g1_s3.mkv --idx 80 --out work/start/real/P0a_g1_s3_f80.png \
      --mask "woman in a red coat sitting on a bench"

  # pinned-beat source: frame 28 of a take, registered into the fixed view (ground-plane homography)
  python scripts/take_frame.py work/beats/A14/g1/A14_g1_s6.mkv --idx 28 --reg --out work/keyframes/real/A14/s6_f28_reg.png

--mask queues sam3_instance.py on the running worker (SAM3 needs the GPU) and waits; the mask is written next to the
frame as <out stem>_mask.png.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from assemble import read_frames  # noqa: E402
from sg_paths import ROOT, SAM3_PY, SCRIPTS  # noqa: E402


def queue_mask(image, prompt, out):
    q = os.path.join(ROOT, "work", "queue")
    os.makedirs(q, exist_ok=True)
    n = f"mask_{os.path.basename(out)[:-4]}_{int(time.time())}"
    cmd = f"{SAM3_PY} {SCRIPTS}/sam3_instance.py --image {image} --prompt '{prompt}' --out {out}"
    p = os.path.join(q, n + ".json")
    json.dump(dict(name=n, cmd=cmd, priority=-1), open(p + ".tmp", "w"))
    os.rename(p + ".tmp", p)
    print("queued", n, "(needs a running worker)")
    while not os.path.exists(out):
        if any(n in f for f in os.listdir(os.path.join(q, "failed")) if os.path.isdir(os.path.join(q, "failed"))):
            raise SystemExit(f"SAM3 request {n} failed: see work/queue/failed/")
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take")
    ap.add_argument("--idx", type=int, default=80)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reg", action="store_true", help="register to the plate (ground-plane homography)")
    ap.add_argument("--mask", default=None, help="SAM3 text prompt; writes <out stem>_mask.png")
    a = ap.parse_args()
    fr = read_frames(a.take)
    f = fr[min(a.idx, len(fr) - 1)]
    if a.reg:
        import bglock
        plate = cv2.cvtColor(cv2.imread(os.path.join(ROOT, "work", "plates", "plate.png")), cv2.COLOR_BGR2RGB)
        Hs, _ = bglock.ground_homographies(f[None], plate, np.zeros((1,) + f.shape[:2], bool), smooth=0)
        f = cv2.warpPerspective(f, Hs[0], (plate.shape[1], plate.shape[0]), flags=cv2.INTER_CUBIC)
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2BGR))
    print("wrote", out)
    if a.mask:
        queue_mask(out, a.mask, out[:-4] + "_mask.png")
        print("wrote", out[:-4] + "_mask.png")


if __name__ == "__main__":
    main()
