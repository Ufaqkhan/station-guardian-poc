"""Generate candidate stills of the empty GP02 view (Gidea Park, Platform 4) with Z-Image-Turbo.

Stand-in for the real GP02 screenshot, which is not on this machine. Writes one PNG per
(prompt variant, seed) and a labelled contact sheet so the best plate can be picked by eye.

Usage: python make_plate.py --out work/plates --seeds 0 1 2 3
"""
import argparse
import json
import os
import time

import torch
from PIL import Image, ImageDraw

SCENE = (
    "The camera looks along the length of the platform, which recedes into the distance towards the top of the frame. "
    "On the left side of the frame: the railway track with grey ballast and two steel rails, and the platform edge "
    "marked with a painted yellow safety line and a strip of textured tactile paving. "
    "Down the middle of the frame: a wide, empty, grey asphalt platform walkway. "
    "On the right side of the frame, standing with their backs against a metal railing fence that runs along the back "
    "edge of the platform: two identical three-seat benches made of horizontal timber slats with purple-painted metal "
    "armrests and legs. The nearer bench stands in the centre-right of the frame, seen at an angle from its end; the "
    "second bench stands further up the platform. A yellow plastic grit bin stands in the bottom right corner. "
)
LIGHT = "Late afternoon sunshine, low sun casting long hard shadows across the platform surface. "
EMPTY = "The platform is completely empty: no people, no train, no text, no on-screen display, no watermark. "

VARIANTS = {
    "cctv_a": (
        "Real CCTV security camera footage, still frame from a surveillance camera mounted high on a pole, looking "
        "steeply down and along an outdoor suburban railway station platform in east London, England. "
        + SCENE + LIGHT + EMPTY +
        "Wide-angle lens, slightly soft focus, slightly washed-out colours, mild sensor noise, typical 1080p security "
        "camera image quality, photorealistic."
    ),
    "cctv_b": (
        "High-angle surveillance camera view of a British railway station platform on a sunny autumn afternoon, "
        "photographed from about six metres up, looking down the platform. "
        + SCENE + LIGHT + EMPTY +
        "Realistic security camera image, muted natural colours, fine compression noise, deep focus, no people."
    ),
    "cctv_c": (
        "A fixed CCTV camera image of Platform 4 at a small suburban station on the Elizabeth line near Romford, "
        "London. The camera is high up and tilted steeply downwards, looking along the platform. "
        + SCENE + LIGHT + EMPTY +
        "Documentary realism, real place, ordinary concrete and asphalt textures, overhead wires and a lamp post, "
        "security camera quality, slightly desaturated."
    ),
    "cctv_d": (
        "Security camera still, elevated view looking steeply down along a long outdoor railway platform in suburban "
        "England. The platform edge and the railway track run along the left side of the picture, a broad grey "
        "walkway runs up the middle, and on the right two wooden three-seat benches with purple metal armrests stand "
        "with their backs to the railing at the back of the platform, one near the centre-right of the picture and "
        "one further away. A yellow grit bin in the bottom right corner. "
        + LIGHT + EMPTY +
        "Photorealistic, CCTV image quality, natural colours."
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--height", type=int, default=864)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from diffusers import ZImagePipeline

    if args.threads:
        torch.set_num_threads(args.threads)
    pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16).to(args.device)
    json.dump({k: VARIANTS[k] for k in args.variants}, open(os.path.join(args.out, "prompts.json"), "w"), indent=1)

    tiles = []
    for v in args.variants:
        for s in args.seeds:
            g = torch.Generator(args.device).manual_seed(s)
            img = pipe(prompt=VARIANTS[v], height=args.height, width=args.width,
                       num_inference_steps=args.steps, guidance_scale=0.0, generator=g).images[0]
            name = f"{v}_s{s}"
            img.save(os.path.join(args.out, name + ".png"))
            tiles.append((name, img))
            print("saved", name, time.strftime("%H:%M:%S"), flush=True)

    # contact sheet: 4 columns, 512x288 tiles, labelled
    tw, th, cols = 512, 288, 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), "black")
    for i, (name, img) in enumerate(tiles):
        t = img.resize((tw, th), Image.LANCZOS)
        d = ImageDraw.Draw(t)
        d.rectangle([0, 0, 150, 18], fill="black")
        d.text((4, 3), name, fill="yellow")
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    sheet.save(os.path.join(args.out, "contact_sheet.jpg"), quality=92)


if __name__ == "__main__":
    main()
