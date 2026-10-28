"""One-page beat sheet: the middle frame of every beat (colour-corrected, CCTV look) with beat id, time, behaviour
and tier, 6 x 3 grid. Writes output/StoryA_GP02_beat_sheet.jpg (and the twin with --twin).
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
import colorfix  # noqa: E402
import story  # noqa: E402
from assemble import (BG, INK, INK2, INK3, TIER_FILL, TIER_TEXT, cctv_look, font,  # noqa: E402
                      read_frames)

from sg_paths import ROOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin", action="store_true")
    ap.add_argument("--frame", type=int, default=48)
    a = ap.parse_args()
    chain = json.load(open(os.path.join(ROOT, "chain", "chain.json")))
    seq = (story.BEATS[:3] + story.TWIN) if a.twin else story.BEATS
    ids = [b["id"] for b in seq]
    resets = {b for b, r in chain["beats"].items() if str(r.get("start", "")).endswith("_cc.png")}
    D = {} if os.path.exists(os.path.join(ROOT, "work", "master", ("twin" if a.twin else "story") + "_1080.mkv")) \
        else colorfix.blended_offsets(ids, resets)
    cols, tw, th = (4 if a.twin else 6), 400, 225
    rows = (len(seq) + cols - 1) // cols
    cap = 58
    W, H = 40 + cols * (tw + 16) - 16 + 40, 110 + rows * (th + cap + 18) + 20
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)
    title = ("Story A, benign twin (35 s)" if a.twin else "Story A, 18 beats (90 s)")
    d.text((40, 30), "STATION GUARDIAN  ·  " + title, font=font(28, bold=True), fill=INK)
    d.text((40, 70), "Gidea Park Platform 4, camera GP02 (fixed view). Synthetic people composited on the real camera view. "
                     "Middle frame of each 5 s beat (1080p master).", font=font(17), fill=INK2)
    master = os.path.join(ROOT, "work", "master", ("twin" if a.twin else "story") + "_1080.mkv")
    M = read_frames(master) if os.path.exists(master) else None
    for k, b in enumerate(seq):
        if M is not None:
            frame = M[k * 80 + a.frame - 1]
        else:
            fr = read_frames(chain["beats"][b["id"]]["clip"])
            if b["id"] in D:
                fr = colorfix.correct_with(fr, b["id"], D[b["id"]])
            frame = fr[a.frame]
        im = Image.fromarray(cctv_look(frame)).resize((tw, th), Image.LANCZOS)
        x = 40 + (k % cols) * (tw + 16)
        y = 110 + (k // cols) * (th + cap + 18)
        sheet.paste(im, (x, y))
        dd = ImageDraw.Draw(sheet)
        dd.text((x, y + th + 8), f"{b['id']}  {b['t0']:02d}–{b['t0'] + 5:02d} s", font=font(17, bold=True), fill=INK)
        dd.text((x, y + th + 32), b["behaviour"], font=font(15), fill=INK3)
        pf = font(15, bold=True)
        pw = dd.textlength(b["tier"], font=pf) + 16
        dd.rounded_rectangle([x + tw - pw, y + th + 8, x + tw, y + th + 30], radius=6, fill=TIER_FILL[b["tier"]])
        dd.text((x + tw - pw + 8, y + th + 10), b["tier"], font=pf, fill=TIER_TEXT[b["tier"]])
    out = os.path.join(ROOT, "output", "StoryA_twin_GP02_beat_sheet.jpg" if a.twin else "StoryA_GP02_beat_sheet.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out, quality=92)
    print(out)


if __name__ == "__main__":
    main()
