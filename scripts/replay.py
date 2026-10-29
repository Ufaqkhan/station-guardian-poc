"""Re-queue the exact generation request behind a kept take of the delivered PoC (records/kept_takes.json).

Each record holds what the worker received: start image, end image (pinned beats), prompt, seed, sigmas, guidance,
frame count. Paths are stored as {ROOT}/... and resolved against the data folder.

  python scripts/replay.py A06                 # -> work/beats/A06/replay/A06_replay_s1.mkv
  python scripts/replay.py A06 A07 --seeds 1 2 3 4   # same request, other seeds (new takes to choose from)
  python scripts/replay.py --list              # beat, seed, mode, start/end frames

The start image of beat N is the corrected last frame of beat N-1 (chain/<N-1>_last_cc.png), so a beat can be replayed
once the beat before it is in chain/ (data bundle 'chain', or after committing it). GPU kernels are not bit-exact
across machines, so a replay matches the kept take closely but not pixel for pixel.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from sg_paths import REPO, ROOT  # noqa: E402

RUNTIME = ("finished", "frames", "gen_seconds", "host", "job", "submitted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("beats", nargs="*")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="default: the kept take's seed")
    ap.add_argument("--tag", default="replay")
    ap.add_argument("--priority", type=int, default=5)
    ap.add_argument("--records", default=os.path.join(REPO, "records", "kept_takes.json"))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    rec = json.loads(open(a.records).read().replace("{ROOT}", ROOT))
    if a.list or not a.beats:
        for b, r in rec.items():
            mode = "first+last" if r.get("last_image") else "first"
            print(f"{b:4s} seed {r['seed']:3d}  {mode:10s} start {os.path.relpath(r['image'], ROOT)}"
                  + (f"  end {os.path.relpath(r['last_image'], ROOT)}" if r.get("last_image") else ""))
        return
    q = os.path.join(ROOT, "work", "queue")
    os.makedirs(q, exist_ok=True)
    for b in a.beats:
        r = {k: v for k, v in rec[b].items() if k not in RUNTIME}
        for k in ("image", "last_image"):
            if r.get(k) and not os.path.exists(r[k]):
                print(f"WARNING {b}: {k} {r[k]} does not exist yet (commit the previous beat / copy the data first)")
        r["out_dir"] = os.path.join(ROOT, "work", "beats", b, a.tag)
        os.makedirs(r["out_dir"], exist_ok=True)
        for s in a.seeds or [rec[b]["seed"]]:
            req = dict(r, seed=s, name=f"{b}_{a.tag}_s{s}", priority=a.priority)
            p = os.path.join(q, req["name"] + ".json")
            json.dump(req, open(p + ".tmp", "w"), indent=1)
            os.rename(p + ".tmp", p)
            print("queued", req["name"], "->", os.path.relpath(req["out_dir"], ROOT))


if __name__ == "__main__":
    main()
