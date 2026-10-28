"""Per-frame person masks with SAM3 (text prompt "person"), for the background lock.

Runs in the `sam3` env:
  python sam3_people.py --frames stab.npy --out masks.npz [--idx 80] [--device cpu]
Input: uint8 array (N,H,W,3) of stabilised frames (.npy). Output: npz with `masks` (N,H,W) bool and
`scores` per frame. With --idx, only those frames are segmented (others are empty).
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image

from sg_paths import SAM3_CKPT as CKPT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--idx", type=int, nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--prompts", nargs="+", default=["person"])
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    fr = np.load(args.frames, mmap_mode="r")
    n, h, w, _ = fr.shape
    idx = args.idx if args.idx else list(range(n))
    idx = [i if i >= 0 else n + i for i in idx]
    if args.device == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    model = build_sam3_image_model(checkpoint_path=CKPT, load_from_HF=False, device=args.device)
    proc = Sam3Processor(model, confidence_threshold=args.conf, device=args.device)
    masks = np.zeros((n, h, w), bool)
    counts = np.zeros(n, np.int16)
    t0 = time.time()
    for i in idx:
        st = proc.set_image(Image.fromarray(np.ascontiguousarray(fr[i])))
        for p in args.prompts:
            proc.reset_all_prompts(st)
            out = proc.set_text_prompt(state=st, prompt=p)
            m = out["masks"].cpu().numpy().astype(bool)
            if m.ndim == 4:
                m = m[:, 0]
            sc = out["scores"].float().cpu().numpy().reshape(-1)
            for k in range(len(sc)):
                if sc[k] >= args.conf and m[k].sum() > 150:
                    masks[i] |= m[k]
                    counts[i] += 1
    np.savez_compressed(args.out, masks=masks, counts=counts, idx=np.array(idx))
    print(f"segmented {len(idx)} frames in {time.time() - t0:.1f}s; people per frame: {counts[idx].tolist()[:12]}...")


if __name__ == "__main__":
    main()
