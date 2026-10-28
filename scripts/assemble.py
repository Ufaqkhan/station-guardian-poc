"""Assemble the selected beat takes into the 90 s deliverables.

Inputs: chain/chain.json  {"fps": 16, "beats": {"A01": {"clip": ".../A01_locked.mkv"}, ...}}
Each take has 81 frames; frame 0 repeats the previous beat's last frame, so frames 1..80 are used
(80 frames = 5.0 s per beat, 18 beats = 1440 frames = 90.0 s).

Outputs (output/):
  StoryA_GP02_clean.mp4      1280x720, the synthetic camera picture only
  StoryA_GP02_cctv.mp4       same with a CCTV on-screen display (camera id, clock, SYNTHETIC tag)
  StoryA_GP02_annotated.mp4  1920x1080 presentation version: CCTV view + beat card + tier timeline

  python scripts/assemble.py                  # all three
  python scripts/assemble.py --preview 37.5   # one annotated still at t=37.5 s (layout check)
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
import story  # noqa: E402

from sg_paths import FONT_DIR, ROOT  # noqa: E402
FPS = 16
FR_PER_BEAT = 80


def font(size, bold=False, mono=False):
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    return ImageFont.truetype(os.path.join(FONT_DIR, name + ("-Bold" if bold else "") + ".ttf"), size)


# ---- palette (dark surface); tier ramp validated as an ordinal one-hue ramp on #16181c ----
BG = (22, 24, 28)
PANEL = (30, 33, 39)
INK = (240, 239, 236)
INK2 = (176, 181, 191)
INK3 = (128, 134, 145)
RULE = (52, 56, 64)
TIER_FILL = {"T0": "#fbe1dc", "T1": "#f7b5a7", "T2": "#f2846e", "T3": "#e1543c", "T4": "#c12b11"}
TIER_TEXT = {"T0": "#16181c", "T1": "#16181c", "T2": "#16181c", "T3": "#ffffff", "T4": "#ffffff"}
TIER_NAME = {"T0": "ordinary activity", "T1": "low-level concern", "T2": "concerning or intimidating",
             "T3": "harassment, confrontation or unwanted touching", "T4": "an assault attempt or violence"}


def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------- video io
def read_frames(path):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    w, h = map(int, probe.stdout.strip().split(","))
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


class Writer:
    def __init__(self, path, w, h, crf=14):
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", path], stdin=subprocess.PIPE)

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


# ---------------------------------------------------------------- CCTV look + on-screen display
_rng = np.random.default_rng(7)


def cctv_look(frame, blur=0.0, desat=0.10, contrast=0.95, lift=4.0, noise=1.2):
    """Subtle security-camera rendering: softer optics, flatter colour, lifted blacks, sensor noise."""
    import cv2

    f = frame.astype(np.float32)
    if blur > 0:
        f = cv2.GaussianBlur(f, (0, 0), blur)
    g = f.mean(-1, keepdims=True)
    f = g + (f - g) * (1 - desat)
    f = (f - 128.0) * contrast + 128.0 + lift
    f += _rng.normal(0, noise, f.shape[:2])[..., None]
    return f.clip(0, 255).astype(np.uint8)


def osd(frame, t, start_clock="2026-09-29 13:36:34"):
    """The GP02 camera's own on-screen display, reproduced: a timestamp in light-grey monospace on a translucent dark
    box at the top right (measured on the real frame: box 209x25 px at 1600 wide, overlay = 0.32 x scene + 10).
    A small SYNTHETIC tag in the same style sits bottom right."""
    W, H = frame.shape[1], frame.shape[0]
    k = W / 1600.0
    clock = dt.datetime.strptime(start_clock, "%Y-%m-%d %H:%M:%S") + dt.timedelta(seconds=int(t))
    text = clock.strftime("%b %d %Y %I:%M:%S %p") + " BST"
    f = font(max(10, int(round(12.4 * k))), mono=True)
    out = frame.astype(np.float32).copy()

    def box(x0, y0, x1, y1):
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * 0.32 + 10.0

    img_tmp = Image.new("RGB", (10, 10))
    tw = ImageDraw.Draw(img_tmp).textlength(text, font=f)
    bw, bh = int(tw + 10 * k), int(round(25 * k))
    box(W - bw, 0, W, bh)
    tag = "SYNTHETIC"
    tw2 = ImageDraw.Draw(img_tmp).textlength(tag, font=f)
    bw2 = int(tw2 + 10 * k)
    box(W - bw2, H - bh, W, H)
    img = Image.fromarray(out.round().clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.text((W - bw + 5 * k, bh / 2), text, font=f, fill=(206, 206, 200), anchor="lm")
    d.text((W - bw2 + 5 * k, H - bh / 2), tag, font=f, fill=(230, 200, 60), anchor="lm")
    return np.asarray(img)


# ---------------------------------------------------------------- annotated layout
VX, VY, VW, VH = 32, 84, 1280, 720           # video box
PX0, PX1 = 1344, 1888                        # beat card
TL_X0, TL_X1 = 32, 1888                      # timeline
BAND_Y0, BAND_Y1 = 832, 864
BAR_BASE, BAR_MAX = 990, 104
TOTAL = 90.0
SEQ = list(story.BEATS)            # beats on the timeline (swapped for the benign twin)
SEQ_PANELS = list(story.PANELS)
HEADER = "BTP Scenario 1 \u00b7 Story A: escalation to attempted sexual assault, lone male"


def tx(t):
    return TL_X0 + (TL_X1 - TL_X0) * t / TOTAL


def wrap_lines(d, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if d.textlength(trial, font=f) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def static_layer():
    img = Image.new("RGB", (1920, 1080), BG)
    d = ImageDraw.Draw(img)
    d.text((32, 22), "STATION GUARDIAN", font=font(26, bold=True), fill=INK)
    x = 32 + d.textlength("STATION GUARDIAN", font=font(26, bold=True)) + 16
    d.text((x, 27), "synthetic scenario  ·  proof of concept", font=font(20), fill=INK2)
    right = HEADER
    d.text((1888, 27), right, font=font(20), fill=INK2, anchor="ra")
    d.line([(32, 66), (1888, 66)], fill=RULE, width=1)
    # video frame border
    d.rectangle([VX - 2, VY - 2, VX + VW + 1, VY + VH + 1], outline=RULE, width=2)
    # beat card background
    d.rounded_rectangle([PX0, VY, PX1, VY + VH], radius=10, fill=PANEL)
    # timeline caption under the video
    d.text((VX, VY + VH + 8), "Gidea Park, Platform 4, camera GP02 (real fixed view). Synthetic people composited on the "
           "real camera picture; no audio.", font=font(15), fill=INK3)
    # tier legend (bottom row)
    lx, ly = TL_X0, 1040
    d.text((lx, ly), "TIER", font=font(15, bold=True), fill=INK3)
    lx += 52
    for k in ["T0", "T1", "T2", "T3", "T4"]:
        d.rounded_rectangle([lx, ly + 1, lx + 16, ly + 17], radius=4, fill=TIER_FILL[k])
        lab = f"{k} {TIER_NAME[k]}"
        d.text((lx + 22, ly), lab, font=font(15), fill=INK2)
        lx += 22 + d.textlength(lab, font=font(15)) + 26
    return img


def draw_card(d, beat):
    x0, x1, y = PX0 + 28, PX1 - 28, VY + 26
    lab = font(15, bold=True)
    d.text((x0, y), "BEAT", font=lab, fill=INK3)
    t0 = beat["t0"]
    d.text((x1, y), "TIME", font=lab, fill=INK3, anchor="ra")
    y += 24
    d.text((x0, y), beat["id"], font=font(56, bold=True), fill=INK)
    d.text((x1, y + 14), f"{t0:02d}–{t0 + 5:02d} s", font=font(30, mono=True), fill=INK, anchor="ra")
    y += 84
    d.line([(x0, y), (x1, y)], fill=RULE, width=1)
    y += 18
    d.text((x0, y), "STORYBOARD PANEL", font=lab, fill=INK3)
    y += 26
    for ln in wrap_lines(d, beat["panel"], font(23, bold=True), x1 - x0):
        d.text((x0, y), ln, font=font(23, bold=True), fill=INK)
        y += 31
    y += 16
    d.text((x0, y), "WHAT THE CAMERA SEES", font=lab, fill=INK3)
    y += 26
    for ln in wrap_lines(d, beat["see"], font(21), x1 - x0):
        d.text((x0, y), ln, font=font(21), fill=INK)
        y += 29
    # behaviour + tier pinned to the bottom of the card
    yb = VY + VH - 190
    d.line([(x0, yb), (x1, yb)], fill=RULE, width=1)
    yb += 18
    d.text((x0, yb), "BEHAVIOUR LABEL", font=lab, fill=INK3)
    yb += 26
    d.text((x0, yb), beat["behaviour"], font=font(24, bold=True), fill=INK)
    yb += 48
    d.text((x0, yb), "TIER", font=lab, fill=INK3)
    yb += 24
    tier = beat["tier"]
    pill = f"{tier}  {TIER_NAME[tier]}"
    pf = font(20, bold=True)
    lines = wrap_lines(d, pill, pf, x1 - x0 - 28)
    ph = 18 + 26 * len(lines)
    d.rounded_rectangle([x0, yb, x1, yb + ph], radius=8, fill=TIER_FILL[tier])
    for i, ln in enumerate(lines):
        d.text((x0 + 14, yb + 9 + 26 * i), ln, font=pf, fill=TIER_TEXT[tier])


def draw_timeline(d, t, cur):
    # storyboard panel bands
    cur_panel = None
    for name, a, b in SEQ_PANELS:
        on = a <= t < b or (t >= TOTAL and b == TOTAL)
        x0, x1 = tx(a) + 1, tx(b) - 1
        d.rounded_rectangle([x0, BAND_Y0, x1, BAND_Y1], radius=5, fill=(70, 76, 88) if on else (38, 41, 48))
        f = font(15, bold=on)
        label = name
        while d.textlength(label, font=f) > (x1 - x0 - 14) and len(label) > 4:
            label = label[:-2]
        if label != name:
            label = label.rstrip() + "…"
        d.text((x0 + 8, BAND_Y0 + 7), label, font=f, fill=INK if on else INK3)
        if on:
            cur_panel = name
    # beat bars: height and fill both encode tier; label below
    for b in SEQ:
        lvl = int(b["tier"][1])
        h = 28 + (BAR_MAX - 28) * lvl / 4
        x0, x1 = tx(b["t0"]) + 1, tx(b["t0"] + 5) - 1
        fill = hexrgb(TIER_FILL[b["tier"]])
        future = b["t0"] > t
        if future:
            fill = tuple(int(c * 0.38 + BG[i] * 0.62) for i, c in enumerate(fill))
        d.rounded_rectangle([x0, BAR_BASE - h, x1, BAR_BASE], radius=4, fill=fill)
        if b["id"] == cur:
            d.rounded_rectangle([x0 - 1, BAR_BASE - h - 1, x1 + 1, BAR_BASE + 1], radius=5, outline=INK, width=2)
        tf = font(13, bold=True)
        tc = TIER_TEXT[b["tier"]] if not future else "#8a8f99"
        d.text(((x0 + x1) / 2, BAR_BASE - h + 6), b["tier"], font=tf, fill=tc, anchor="ma")
        d.text(((x0 + x1) / 2, BAR_BASE + 6), b["id"], font=font(13, bold=b["id"] == cur),
               fill=INK if b["id"] == cur else INK3, anchor="ma")
    d.line([(TL_X0, BAR_BASE), (TL_X1, BAR_BASE)], fill=RULE, width=1)
    # playhead
    x = tx(min(t, TOTAL))
    d.line([(x, BAND_Y1 + 6), (x, BAR_BASE + 2)], fill=INK, width=2)
    d.polygon([(x - 6, BAND_Y1 + 4), (x + 6, BAND_Y1 + 4), (x, BAND_Y1 + 12)], fill=INK)
    m, s = divmod(int(t), 60)
    d.text((TL_X1, 1040), f"{m:02d}:{s:02d} / {int(TOTAL) // 60:02d}:{int(TOTAL) % 60:02d}", font=font(16, bold=True, mono=True), fill=INK, anchor="ra")


def annotated(frame_cctv, t, static, card_cache):
    beat = SEQ[min(int(t // 5), len(SEQ) - 1)]
    if beat["id"] not in card_cache:
        im = static.copy()
        draw_card(ImageDraw.Draw(im), beat)
        card_cache[beat["id"]] = im
    im = card_cache[beat["id"]].copy()
    fr = Image.fromarray(frame_cctv)
    if fr.size != (VW, VH):
        fr = fr.resize((VW, VH), Image.LANCZOS)
    im.paste(fr, (VX, VY))
    draw_timeline(ImageDraw.Draw(im), t, beat["id"])
    return np.asarray(im)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default=os.path.join(ROOT, "chain", "chain.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "output"))
    ap.add_argument("--preview", type=float, default=None, help="render one annotated still at this time (s)")
    ap.add_argument("--preview_frame", default=None, help="image to use as the video frame for --preview")
    ap.add_argument("--prefix", default="StoryA_GP02")
    ap.add_argument("--partial", action="store_true", help="assemble the committed beats up to the first missing one")
    ap.add_argument("--no_colorfix", action="store_true")
    ap.add_argument("--twin", action="store_true", help="assemble the benign twin (A01-A03 + AT1-AT4, 35 s)")
    ap.add_argument("--stills", type=float, nargs="*", default=None, help="also save annotated stills at these times")
    ap.add_argument("--write_master", default=None, help="write the colour-fixed 720p frames (lossless) here and stop")
    ap.add_argument("--master", default=None, help="build the deliverables from this master video (e.g. 1080p)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    static = static_layer()

    if args.preview is not None:
        fr = np.asarray(Image.open(args.preview_frame).convert("RGB").resize((VW, VH), Image.LANCZOS))
        out = annotated(osd(cctv_look(fr), args.preview), args.preview, static, {})
        p = os.path.join(args.out, f"preview_t{args.preview:05.1f}.png")
        Image.fromarray(out).save(p)
        print(p)
        return

    chain = json.load(open(args.chain))
    global SEQ, SEQ_PANELS, TOTAL, HEADER
    if args.twin:
        SEQ = list(story.BEATS[:3]) + list(story.TWIN)
        SEQ_PANELS = [("1 Baseline", 0, 15), ("Benign twin: he takes the empty far bench", 15, 35)]
        TOTAL = 35.0
        HEADER = "BTP Scenario 1 \u00b7 Story A, benign twin: same opening, nothing happens"
        static = static_layer()
    ids = [b["id"] for b in SEQ]
    missing = [i for i in ids if i not in chain["beats"]]
    if args.partial and missing:
        ids = ids[: ids.index(missing[0])]
        print("partial assembly:", ids[0], "-", ids[-1])
    else:
        assert not missing, f"beats missing from chain: {missing}"
    w_clean = w_cctv = w_ann = None
    cache = {}
    n = 0

    def chain_frames():
        import colorfix
        resets = {b for b, r in chain["beats"].items() if str(r.get("start", "")).endswith("_cc.png")}
        Dall = colorfix.blended_offsets(ids, resets) if not args.no_colorfix else {}
        print("colour fix on", sorted(Dall), "| reset boundaries before", sorted(resets))
        # each beat was generated from (a version of) the previous beat's last frame, which is the next beat's
        # frame 0; ease the last TAIL frames of a beat into that frame so every boundary is seamless
        TAIL = 8
        prev, prev_id = None, None
        for bid in ids:
            fr = read_frames(chain["beats"][bid]["clip"])
            if bid in Dall:
                fr = colorfix.correct_with(fr, bid, Dall[bid])
            assert len(fr) >= FR_PER_BEAT + 1, (bid, len(fr))
            if prev is not None:
                tgt = fr[0].astype(np.float32)
                for j in range(TAIL):
                    k = FR_PER_BEAT - TAIL + 1 + j
                    w = 0.5 - 0.5 * np.cos(np.pi * (j + 1) / TAIL)
                    prev[k] = ((1 - w) * prev[k].astype(np.float32) + w * tgt).round().clip(0, 255).astype(np.uint8)
                for f in prev[1:FR_PER_BEAT + 1]:
                    yield f
                print(f"{prev_id} done", flush=True)
            prev, prev_id = fr.copy(), bid
        for f in prev[1:FR_PER_BEAT + 1]:
            yield f
        print(f"{prev_id} done", flush=True)

    def master_frames(path):
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
        w, h = map(int, probe.stdout.strip().split(","))
        pr = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                              stdout=subprocess.PIPE)
        while True:
            b = pr.stdout.read(w * h * 3)
            if len(b) < w * h * 3:
                break
            yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
        pr.wait()

    if args.write_master:
        wm = None
        for f in chain_frames():
            if wm is None:
                h, w = f.shape[:2]
                wm = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                       "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "ffv1", "-level", "3",
                                       "-pix_fmt", "bgr0", args.write_master], stdin=subprocess.PIPE)
            wm.stdin.write(np.ascontiguousarray(f).tobytes())
            n += 1
        wm.stdin.close()
        wm.wait()
        json.dump(dict(beats=ids, frames=n, fps=FPS), open(args.write_master + ".json", "w"))
        print(f"master: {n} frames -> {args.write_master}")
        return

    src = master_frames(args.master) if args.master else chain_frames()
    for f in src:
        t = n / FPS
        if w_clean is None:
            h, w = f.shape[:2]
            w_clean = Writer(os.path.join(args.out, f"{args.prefix}_clean.mp4"), w, h, crf=15)
            w_cctv = Writer(os.path.join(args.out, f"{args.prefix}_cctv.mp4"), w, h, crf=18)
            w_ann = Writer(os.path.join(args.out, f"{args.prefix}_annotated.mp4"), 1920, 1080, crf=18)
        c = osd(cctv_look(f), t)
        w_clean.write(f)
        w_cctv.write(c)
        fa = annotated(c, t, static, cache)
        w_ann.write(fa)
        if args.stills and any(abs(t - x) < 0.5 / FPS for x in args.stills):
            Image.fromarray(fa).save(os.path.join(args.out, f"still_t{t:05.1f}.jpg"), quality=92)
        n += 1
    for w in (w_clean, w_cctv, w_ann):
        w.close()
    print(f"wrote {n} frames = {n / FPS:.2f} s")


if __name__ == "__main__":
    main()
