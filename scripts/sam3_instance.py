"""Best SAM3 mask for one text prompt on one image (sam3 env, GPU).
  python sam3_instance.py --image f.png --prompt "man in a green jacket" --out mask.png
"""
import argparse
import numpy as np, torch, cv2
from PIL import Image
from sg_paths import SAM3_CKPT as CKPT  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--image", required=True); ap.add_argument("--prompt", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
proc = Sam3Processor(build_sam3_image_model(checkpoint_path=CKPT, load_from_HF=False), confidence_threshold=0.2)
st = proc.set_image(Image.open(a.image).convert("RGB"))
o = proc.set_text_prompt(state=st, prompt=a.prompt)
m = o["masks"].cpu().numpy().astype(bool); m = m[:, 0] if m.ndim == 4 else m
sc = o["scores"].float().cpu().numpy().reshape(-1)
k = int(sc.argmax()); cv2.imwrite(a.out, (m[k] * 255).astype(np.uint8)); print("score", float(sc[k]), "px", int(m[k].sum()))
