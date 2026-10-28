"""Seated-woman height through a committed beat under ground vs bench registration (SAM3 instance masks).
  python scripts/sizecheck2.py A06 [A07 ...]
"""
import os, sys, json
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import bglock
from assemble import read_frames
from sg_paths import ROOT as R0  # noqa: E402
pl = cv2.cvtColor(cv2.imread(os.path.join(R0, "work/plates/plate.png")), cv2.COLOR_BGR2RGB)
c = json.load(open(os.path.join(R0, "chain/chain.json")))
for b in sys.argv[1:]:
    raw = read_frames(c["beats"][b]["take"])
    inst = np.load(os.path.join(R0, "work/lock", b, "instances.npz"))["labels"]
    ppl = np.load(os.path.join(R0, "work/lock", b, "people.npz"))["masks"]
    idx = list(range(0, 81, 10))
    out = {}
    for name, fn in (("ground", bglock.ground_homographies), ("bench", bglock.bench_homographies)):
        Hs, _ = fn(raw, pl, ppl)
        hs = []
        for i in idx:
            lab = inst[i]; best = None
            for k in range(1, int(lab.max()) + 1):
                m = lab == k
                if m.sum() < 800: continue
                red = ((raw[i][..., 0].astype(int) - raw[i][..., 1]) > 70)[m].mean()
                if best is None or red > best[0]: best = (red, m)
            if best is None: hs.append(np.nan); continue
            w = cv2.warpPerspective(best[1].astype(np.uint8), Hs[i], (1280, 720), flags=cv2.INTER_NEAREST)
            ys = np.nonzero(w)[0]; hs.append(ys.max() - ys.min() if len(ys) else np.nan)
        out[name] = np.array(hs, float)
    for name, h in out.items():
        print(f"{b} {name:6s} woman height rel f0:", " ".join(f"{x / h[0]:.2f}" for x in h), f"(f0 {h[0]:.0f}px)")
