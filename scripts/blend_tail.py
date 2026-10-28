"""Make a regenerated, end-pinned beat hand over exactly to the untouched next beat.

The next beat was generated from the OLD version of this beat's last frame. The new take was generated with that
frame as its last-frame target, so it ends close to it but not exactly; this blends the last `n` frames of the new
(locked) clip into the old (locked) clip's frames, ending on the old last frame pixel for pixel.

  python scripts/blend_tail.py A05 work/lock/pre_legs/A05.mkv [--n 8]
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import bglock  # noqa: E402
from assemble import read_frames  # noqa: E402

from sg_paths import ROOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("beat")
    ap.add_argument("old_clip")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    clip = os.path.join(ROOT, "chain", f"{a.beat}.mkv")
    new = read_frames(clip).astype(np.float32)
    old = read_frames(a.old_clip).astype(np.float32)
    end = len(new) - 1
    print(f"{a.beat}: before blend |new f{end} - old f{end}| = {np.abs(new[end] - old[end]).mean():.2f}, "
          f"typical frame step {np.abs(np.diff(new[-10:], axis=0)).mean():.2f}")
    for k in range(end - a.n + 1, end + 1):
        w = (k - (end - a.n)) / a.n            # 1/n ... 1
        w = 0.5 - 0.5 * np.cos(np.pi * w)      # smooth ease-in
        new[k] = (1 - w) * new[k] + w * old[k]
    out = new.round().clip(0, 255).astype(np.uint8)
    bglock.write_ffv1(out, clip)
    cv2.imwrite(os.path.join(ROOT, "chain", f"{a.beat}_last.png"), cv2.cvtColor(out[-1], cv2.COLOR_RGB2BGR))
    d = np.abs(np.diff(out[-a.n - 2:].astype(np.float32), axis=0)).mean(axis=(1, 2, 3))
    print("frame steps over the blend:", np.round(d, 2).tolist(), "| last frame == old:", bool((out[-1] == old[-1].astype(np.uint8)).all()))


if __name__ == "__main__":
    main()
