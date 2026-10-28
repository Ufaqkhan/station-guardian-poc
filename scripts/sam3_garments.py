"""Per-garment colour statistics with SAM3 text prompts (runs in the `sam3` env, on the worker GPU).

  python sam3_garments.py --frames frames.npy --idx 0 20 40 60 80 --out stats.json

For each requested frame and each garment prompt: pixel count and median Lab (OpenCV 8-bit Lab) of the
highest-scoring mask, eroded a little so edges and shadows do not dominate. Used by colorfix.py to keep the
two people's colours stable along the chain of generated beats.
"""
import argparse
import json

import cv2
import numpy as np
import torch
from PIL import Image

from sg_paths import SAM3_CKPT as CKPT  # noqa: E402
ALT = {"short grey hair": ["short grey hair", "dark blue hat", "man's head"]}   # fallbacks when the hair has drifted
GARMENTS = ["green puffer jacket", "red coat", "jeans", "khaki trousers", "white knitted beanie", "short grey hair", "white trainers", "black handbag", "head of the man in the green jacket"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--idx", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--labels", default=None, help="also save a per-frame garment label map (npz, uint8, 0=none)")
    args = ap.parse_args()
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    model = build_sam3_image_model(checkpoint_path=CKPT, load_from_HF=False)
    proc = Sam3Processor(model, confidence_threshold=args.conf)
    fr = np.load(args.frames, mmap_mode="r")
    k = np.ones((5, 5), np.uint8)
    res = {}
    labels = np.zeros((len(args.idx),) + fr.shape[1:3], np.uint8) if args.labels else None
    for li, i in enumerate(args.idx):
        img = np.ascontiguousarray(fr[i])
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        st = proc.set_image(Image.fromarray(img))
        res[str(i)] = {}
        for gi, g in enumerate(GARMENTS):
            best = None
            for pr in ALT.get(g, [g]):
                proc.reset_all_prompts(st)
                out = proc.set_text_prompt(state=st, prompt=pr)
                m = out["masks"].cpu().numpy().astype(bool)
                if m.ndim == 4:
                    m = m[:, 0]
                sc = out["scores"].float().cpu().numpy().reshape(-1)
                if len(sc) and sc.max() >= args.conf:
                    cand = m[int(sc.argmax())]
                    if pr == "man's head":      # keep only the top 40 % of the head box (hair / hat)
                        ys, xs = np.where(cand)
                        if len(ys):
                            cand = cand & (np.arange(cand.shape[0])[:, None] < ys.min() + 0.4 * (ys.max() - ys.min()))
                    best = (cand, float(sc.max()))
                    break
            if best is None:
                continue
            full, scmax = best
            sc = np.array([scmax])
            if labels is not None:
                labels[li][full & (labels[li] == 0)] = gi + 1
            mm = cv2.erode(full.astype(np.uint8), k) > 0
            if mm.sum() < 200:
                continue
            px = lab[mm].astype(np.float32)
            res[str(i)][g] = dict(n=int(mm.sum()), med=np.median(px, 0).tolist(), score=float(sc.max()))
    json.dump(res, open(args.out, "w"), indent=1)
    if labels is not None:
        np.savez_compressed(args.labels, labels=labels, garments=np.array(GARMENTS))
    print("garment stats for", len(args.idx), "frames")


if __name__ == "__main__":
    main()
