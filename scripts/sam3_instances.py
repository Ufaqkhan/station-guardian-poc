"""Per-person instance masks with SAM3 for every frame of a take (sam3 env, GPU).

  python sam3_instances.py --video take.mkv --out instances.npz

Output: `labels` uint8 (N, H, W): 0 = background, k = person k in that frame (ordered by score; a pixel claimed by two
instances goes to the higher score). Handbag pixels are given to the person whose mask they touch most.
`scores` float (N, 8). Coordinates are those of the take (not yet registered to the plate).
"""
import argparse

import cv2
import numpy as np
import torch
from PIL import Image

from sg_paths import SAM3_CKPT as CKPT  # noqa: E402


def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def masks_for(proc, st, prompt, conf):
    proc.reset_all_prompts(st)
    out = proc.set_text_prompt(state=st, prompt=prompt)
    m = out["masks"].cpu().numpy().astype(bool)
    if m.ndim == 4:
        m = m[:, 0]
    sc = out["scores"].float().cpu().numpy().reshape(-1)
    keep = [(sc[k], m[k]) for k in range(len(sc)) if sc[k] >= conf and m[k].sum() > 300]
    return sorted(keep, key=lambda t: -t[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.35)
    a = ap.parse_args()
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    proc = Sam3Processor(build_sam3_image_model(checkpoint_path=CKPT, load_from_HF=False), confidence_threshold=a.conf)
    fr = read_video(a.video)
    n, h, w, _ = fr.shape
    labels = np.zeros((n, h, w), np.uint8)
    scores = np.zeros((n, 8), np.float32)
    for i in range(n):
        st = proc.set_image(Image.fromarray(fr[i]))
        people = masks_for(proc, st, "person", a.conf)[:8]
        for k, (s, m) in enumerate(people):
            labels[i][m & (labels[i] == 0)] = k + 1
            scores[i, k] = s
        for s, m in masks_for(proc, st, "handbag", 0.3):
            if not people:
                break
            ring = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            touch = [int((ring & (labels[i] == k + 1)).sum()) for k in range(len(people))]
            if max(touch) > 0:
                labels[i][m & (labels[i] == 0)] = int(np.argmax(touch)) + 1
    np.savez_compressed(a.out, labels=labels, scores=scores)
    print("instances:", n, "frames; people per frame", sorted(set((scores > 0).sum(1).tolist())))


if __name__ == "__main__":
    main()
