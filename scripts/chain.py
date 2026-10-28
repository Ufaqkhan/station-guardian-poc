"""Chain bookkeeping and review sheets.

  python chain.py init --plate work/plates/plate.png --start work/start/A01_start.png
  python chain.py review A04 r1            # zoomed review sheet per take: work/beats/A04/r1/<take>_zoom.jpg
  python chain.py choose A04 work/beats/A04/r1/A04_r1_s3.mkv [--no_lock]
  python chain.py status

chain/chain.json keeps: plate, crop box for review, and per beat the chosen take, its lock stats and prompt.
chain/<beat>.mkv is the (background-locked) take; chain/<beat>_last.png is the next beat's start image.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from assemble import read_frames  # noqa: E402
import bglock  # noqa: E402
import story  # noqa: E402

from sg_paths import ROOT  # noqa: E402
CH = os.path.join(ROOT, "chain")
CJ = os.path.join(CH, "chain.json")


def load():
    return json.load(open(CJ)) if os.path.exists(CJ) else {"fps": 16, "beats": {}}


def save(c):
    os.makedirs(CH, exist_ok=True)
    json.dump(c, open(CJ + ".tmp", "w"), indent=1)
    os.replace(CJ + ".tmp", CJ)


def prev_last(beat_id):
    ids = [b["id"] for b in story.BEATS]
    c = load()
    if beat_id == "A01":
        return c["start"]
    if beat_id in ("AT1",):
        return os.path.join(CH, "A03_last.png")
    if beat_id.startswith("AT"):
        return os.path.join(CH, f"AT{int(beat_id[2:]) - 1}_last.png")
    return os.path.join(CH, f"{ids[ids.index(beat_id) - 1]}_last.png")


def zoom_sheet(mkv, box, out, idx=(0, 16, 32, 48, 64, 80)):
    fr = read_frames(mkv)
    x0, y0, x1, y1 = box
    tiles = []
    for i in idx:
        if i >= len(fr):
            continue
        t = cv2.cvtColor(np.ascontiguousarray(fr[i, y0:y1, x0:x1]), cv2.COLOR_RGB2BGR)
        cv2.putText(t, f"f{i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(t)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[r:r + 3], 1) for r in range(0, len(tiles), 3)]
    cv2.imwrite(out, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd")
    a = sp.add_parser("init")
    a.add_argument("--plate", required=True)
    a.add_argument("--start", required=True)
    a.add_argument("--zoom", type=int, nargs=4, default=None, help="review crop x0 y0 x1 y1 (1280x720 px)")
    a = sp.add_parser("review")
    a.add_argument("beat")
    a.add_argument("tag")
    a.add_argument("--box", type=int, nargs=4, default=None)
    a = sp.add_parser("choose")
    a.add_argument("beat")
    a.add_argument("take")
    a.add_argument("--no_lock", action="store_true")
    a.add_argument("--thr", type=float, default=16.0)
    sp.add_parser("status")
    args = ap.parse_args()

    c = load()
    if args.cmd == "init":
        c.update(plate=os.path.abspath(args.plate), start=os.path.abspath(args.start))
        if args.zoom:
            c["zoom"] = args.zoom
        save(c)
        print(json.dumps(c, indent=1))
    elif args.cmd == "review":
        box = args.box or c.get("zoom", [640, 120, 1280, 720])
        for mkv in sorted(glob.glob(os.path.join(ROOT, "work", "beats", args.beat, args.tag, "*.mkv"))):
            if mkv.endswith("_mask.mkv"):
                continue
            out = mkv[:-4] + "_zoom.jpg"
            zoom_sheet(mkv, box, out)
            print(out)
    elif args.cmd == "choose":
        os.makedirs(CH, exist_ok=True)
        fr = read_frames(args.take)
        plate = cv2.cvtColor(cv2.imread(c["plate"]), cv2.COLOR_BGR2RGB)
        plate = cv2.resize(plate, (fr.shape[2], fr.shape[1]), interpolation=cv2.INTER_AREA)
        out = os.path.join(CH, f"{args.beat}.mkv")
        rec = dict(take=os.path.abspath(args.take))
        if args.no_lock:
            bglock.write_ffv1(fr, out)
            last = fr[-1]
        else:
            locked, m, cover, skipped = bglock.lock(fr, plate, thr=args.thr)
            bglock.write_ffv1(locked, out)
            last = locked[-1]
            rec.update(mask_cover_mean=round(float(cover.mean()), 3), mask_cover_max=round(float(cover.max()), 3),
                       lock_skipped=int(skipped))
        cv2.imwrite(os.path.join(CH, f"{args.beat}_last.png"), cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
        meta = args.take[:-4] + ".json"
        if os.path.exists(meta):
            m = json.load(open(meta))
            rec.update(prompt=m.get("prompt"), seed=m.get("seed"), start=m.get("image"))
        rec["clip"] = out
        c["beats"][args.beat] = rec
        save(c)
        print(json.dumps(rec, indent=1))
    elif args.cmd == "status":
        for b in story.BEATS + story.TWIN:
            r = c["beats"].get(b["id"])
            print(b["id"], "-" if not r else os.path.basename(r["take"]), "" if not r else r.get("mask_cover_mean"))


if __name__ == "__main__":
    main()
