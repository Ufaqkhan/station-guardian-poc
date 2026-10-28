"""Inpaint into the chosen plate with Z-Image-Turbo (runs on CPU or GPU).

Used for (1) the A01 opening frame: the young woman seated on the middle seat of the near bench, and
(2) cleaning the plate (e.g. removing an object). The result outside the mask is taken back from the
plate pixel-for-pixel, so only the masked area changes.

  python make_start.py --plate plate.png --box x0 y0 x1 y1 --prompt woman --seeds 0 1 2 --out work/start/r1
"""
import argparse
import os
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

PROMPTS = {
    "woman": (
        "High-angle CCTV security camera still of a railway station platform on a sunny afternoon. A young woman in "
        "her early twenties sits alone on the middle seat of the wooden bench with purple armrests, facing left "
        "towards the railway track, her head bowed, looking down at the phone in her hands, a small black cross-body "
        "bag on her lap. She is slim, with dark hair in a low ponytail under a white knitted beanie, and wears a "
        "knee-length red coat, black jeans and white trainers. Low afternoon sun, she casts a soft long shadow. "
        "Photorealistic, natural colours, security camera image quality."
    ),
    "woman_mid": (
        "High-angle CCTV security camera still looking down along a railway station platform on a sunny afternoon. "
        "A young woman in her early twenties sits alone in the middle of the long wooden bench with purple armrests, "
        "her back against the wooden backrest, facing left towards the railway track, head bowed over the phone in "
        "her hands, a small black cross-body bag on her lap, feet on the ground in front of the bench. She is slim, "
        "with dark hair in a low ponytail under a white knitted beanie, and wears a knee-length red coat, black jeans "
        "and white trainers. Her whole body is visible, head to feet. Low afternoon sun from behind the bench, her "
        "shadow falls on the platform in front of her. Photorealistic, natural colours, security camera image quality."
    ),
    "woman_proxy": (
        "High-angle photo looking down at a wooden bench with purple armrests on a railway station platform on a sunny "
        "afternoon. A young woman in her early twenties sits on the bench, facing left, head bowed over the phone in "
        "her hands, a small black cross-body bag on her lap, her feet in white trainers on the ground in front of the "
        "bench. She is slim, with dark hair in a low ponytail under a white knitted beanie, and wears a knee-length "
        "red wool coat and black jeans. Realistic proportions, photorealistic, natural light and shadow."
    ),
    "woman_legs": (
        "High-angle photo of a young woman sitting on a wooden bench on a sunny railway platform: her lower legs in "
        "slim black jeans go straight down from her knees to the ground, and her feet in white trainers rest flat on "
        "the grey asphalt in front of the bench, casting a soft shadow. Photorealistic, natural light, sharp detail."
    ),
    "clean": (
        "Grey metal railing fence along the back of a railway platform, a brick building wall and a railway track "
        "behind it, afternoon sunlight. Photorealistic security camera image, no camera, no objects."
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", required=True)
    ap.add_argument("--box", type=int, nargs=4, action="append", default=None, help="mask box x0 y0 x1 y1 (plate px)")
    ap.add_argument("--init", default=None, help="image to inpaint from (e.g. plate + stand-in); default = plate")
    ap.add_argument("--mask", default=None, help="mask png instead of boxes")
    ap.add_argument("--ellipse", action="store_true", help="elliptical mask inside each box")
    ap.add_argument("--prompt", default="woman")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--feather", type=float, default=6.0)
    ap.add_argument("--name", default=None, help="output file stem (default: prompt key)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(args.threads)

    from diffusers import ZImageInpaintPipeline

    plate = Image.open(args.plate).convert("RGB")
    W, H = plate.size
    init = Image.open(args.init).convert("RGB") if args.init else plate
    if args.mask:
        mask = Image.open(args.mask).convert("L")
    else:
        mask = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(mask)
        for b in args.box:
            (d.ellipse if args.ellipse else d.rectangle)(b, fill=255)
    mask.save(os.path.join(args.out, "mask.png"))
    soft = mask.filter(ImageFilter.GaussianBlur(args.feather))

    pipe = ZImageInpaintPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16).to(args.device)
    prompt = PROMPTS.get(args.prompt, args.prompt)
    for s in args.seeds:
        g = torch.Generator(args.device).manual_seed(s)
        img = pipe(prompt=prompt, image=init, mask_image=mask, strength=args.strength, height=H, width=W,
                   num_inference_steps=args.steps, guidance_scale=0.0, generator=g).images[0]
        # outside the mask: exact plate pixels
        a = np.asarray(soft, np.float32)[..., None] / 255.0
        comp = (np.asarray(img, np.float32) * a + np.asarray(plate, np.float32) * (1 - a)).round().astype(np.uint8)
        stem = args.name or args.prompt
        Image.fromarray(comp).save(os.path.join(args.out, f"{stem}_st{args.strength:.2f}_s{s}.png"))
        print("saved", s, time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
