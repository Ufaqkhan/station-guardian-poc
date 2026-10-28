"""Remove the camera's burned-in timestamp (semi-transparent box, top-right) from the real GP02 frame.

Only a 512x256 patch around the box is inpainted with Z-Image-Turbo; outside the box mask every pixel is the original.
  python scripts/clean_osd.py --seeds 1 2 3
Writes work/plates/real/clean_s<seed>.png (1600x900) and a zoom sheet.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from sg_paths import ROOT  # noqa: E402
BOX = (1386, 0, 1600, 27)            # timestamp box in the 1600x900 frame
PATCH = (1088, 0, 1600, 256)
PROMPT = ("High-angle CCTV still of a sunny railway station platform seen from above: grey asphalt and paving, a "
          "painted yellow safety line, white and grey metal posts, a black metal railing, sunlight and shadows. "
          "Photorealistic security camera image, no text, no numbers, no overlay.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--threads", type=int, default=48)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from diffusers import ZImageInpaintPipeline

    src = Image.open(os.path.join(ROOT, "work", "plates", "real", "view_original.png")).convert("RGB")
    patch = src.crop(PATCH)
    m = Image.new("L", patch.size, 0)
    x0, y0, x1, y1 = BOX[0] - PATCH[0], BOX[1] - PATCH[1], BOX[2] - PATCH[0], BOX[3] - PATCH[1]
    ImageDraw.Draw(m).rectangle([x0 - 3, y0, x1, y1 + 3], fill=255)
    soft = np.asarray(m.filter(ImageFilter.GaussianBlur(2)), np.float32)[..., None] / 255.0
    pipe = ZImageInpaintPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16).to("cpu")
    tiles = []
    for s in a.seeds:
        g = torch.Generator("cpu").manual_seed(s)
        out = pipe(prompt=PROMPT, image=patch, mask_image=m, strength=1.0, height=patch.size[1], width=patch.size[0],
                   num_inference_steps=9, guidance_scale=0.0, generator=g).images[0]
        comp = (np.asarray(out, np.float32) * soft + np.asarray(patch, np.float32) * (1 - soft)).round().astype(np.uint8)
        full = src.copy()
        full.paste(Image.fromarray(comp), PATCH[:2])
        full.save(os.path.join(ROOT, "work", "plates", "real", f"clean_s{s}.png"))
        tiles.append(Image.fromarray(comp).crop((260, 0, 512, 64)).resize((756, 192), Image.LANCZOS))
        print("saved", s, flush=True)
    sheet = Image.new("RGB", (756, 192 * (len(tiles) + 1)))
    sheet.paste(patch.crop((260, 0, 512, 64)).resize((756, 192), Image.LANCZOS), (0, 0))
    for i, t in enumerate(tiles):
        sheet.paste(t, (0, 192 * (i + 1)))
    sheet.save(os.path.join(ROOT, "work", "plates", "real", "clean_sheet.jpg"), quality=92)


if __name__ == "__main__":
    main()
