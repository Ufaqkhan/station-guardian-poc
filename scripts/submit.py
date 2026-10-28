"""Queue Wan2.2 generation requests for one beat (one request per seed).

  python scripts/submit.py --beat A04 --start chain/A03_last.png --seeds 1 2 3 4
  python scripts/submit.py --beat P0a --start work/plates/plate.png --seeds 1 2 3 4 --tag r1
  python scripts/submit.py --beat A14 --start ... --prompt "custom text"   # override the story prompt
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import story  # noqa: E402

from sg_paths import ROOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beat", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--extra", default="", help="text appended to the story prompt")
    ap.add_argument("--last", default=None, help="optional end frame (first-last-frame mode)")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25])
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--guidance2", type=float, default=None, help="low-noise expert guidance (default = --guidance)")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--priority", type=int, default=5)
    ap.add_argument("--negative", default=None, help="negative prompt (only matters with --guidance > 1)")
    args = ap.parse_args()

    if args.prompt:
        prompt = args.prompt
    elif args.beat in story.PREROLL:
        prompt = story.PREROLL[args.beat]
    else:
        prompt = story.ALL[args.beat]["prompt"]
    prompt = (prompt + " " + args.extra).strip()
    out_dir = os.path.join(ROOT, "work", "beats", args.beat, args.tag)
    q = os.path.join(ROOT, "work", "queue")
    os.makedirs(q, exist_ok=True)
    for s in args.seeds:
        req = dict(name=f"{args.beat}_{args.tag}_s{s}", image=os.path.abspath(args.start), prompt=prompt, seed=s,
                   out_dir=out_dir, sigmas=args.sigmas, guidance=args.guidance, num_frames=args.frames,
                   last_image=os.path.abspath(args.last) if args.last else None, priority=args.priority,
                   **({"negative_prompt": args.negative} if args.negative else {}),
                   **({"guidance2": args.guidance2} if args.guidance2 is not None else {}),
                   submitted=time.strftime("%Y-%m-%d %H:%M:%S"))
        tmp = os.path.join(q, f".{req['name']}.tmp")
        json.dump(req, open(tmp, "w"), indent=1)
        os.rename(tmp, os.path.join(q, req["name"] + ".json"))
        print("queued", req["name"])


if __name__ == "__main__":
    main()
