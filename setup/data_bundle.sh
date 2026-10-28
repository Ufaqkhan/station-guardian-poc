#!/bin/bash
# Pack the data that is NOT in git (real camera plate, opening frame, kept takes, ...) on the OLD server, so it can be
# copied to a new one. Run it where the data lives; SG_ROOT (config/env.sh) or --root says where that is.
#
#   setup/data_bundle.sh list  base|chain|full [--root DIR]      # print the file list (relative to the root)
#   setup/data_bundle.sh tar   base|chain|full OUT.tar [--root DIR]
#
# Tiers (sizes for Story A on the original server):
#   base   ~10 MB   camera plate (+1536 px, SeedVR2-upscaled), bench occlusion mask, opening frame, colour reference,
#                   chain.json. Enough to generate NEW beats/stories on the GP02 view.
#   chain  ~4.3 GB  base + every kept beat (chain/), its raw take, lock data (masks, homographies, mattes), colour
#                   offsets. Enough to re-compose, re-colour and re-render Story A, or continue from any beat.
#   full   ~11.5 GB chain + 720p/1080p masters (work/master) + delivered videos (output/).
#
# On the new server:  tar -xf OUT.tar -C "$SG_ROOT"  &&  $SG_PY setup/relocate_paths.py --old "$(cat "$SG_ROOT/OLD_ROOT.txt")"
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config/load_env.sh" 2>/dev/null
MODE=${1:?list|tar}; TIER=${2:?base|chain|full}; shift 2
OUT=""; [ "$MODE" = tar ] && { OUT=$(realpath -m "${1:?OUT.tar}"); shift; }
[ "${1:-}" = "--root" ] && SG_ROOT=$(realpath "$2")
cd "$SG_ROOT"

files() {
  for f in work/plates/plate.png work/plates/plate_1536.png work/plates/real/bench_front_mask.png \
           work/start/real/A01_start.png work/colorfix/reference.json \
           work/colorfix/ref_A01_stats.json work/colorfix/ref_A05_stats.json chain/chain.json; do
    [ -e "$f" ] && echo "$f"
  done
  [ -d work/plates/plate_sr ] && find work/plates/plate_sr -type f
  if [ "$TIER" != base ]; then
    find chain -maxdepth 1 -type f \( -name "*.mkv" -o -name "*.png" -o -name "*.json" \) | grep -v "chain/chain.json"
    # the raw take of every kept beat (+ its request json), as recorded in chain.json
    "$SG_PY" - <<'PY'
import json, os
root = os.getcwd()
for b, r in json.load(open("chain/chain.json"))["beats"].items():
    t = r.get("take", "")
    for p in (t, t[:-4] + ".json"):
        rel = os.path.relpath(p, root) if os.path.isabs(p) else p
        if p and os.path.exists(rel):
            print(rel)
PY
    find work/lock work/colorfix work/keyframes -type f \( -name "*.npz" -o -name "*.npy" -o -name "*.json" -o -name "*.png" \)
  fi
  if [ "$TIER" = full ]; then
    find work/master output -type f
  fi
}

case "$MODE" in
  list) files | sort -u ;;
  tar)
    echo "$PWD" > OLD_ROOT.txt
    { echo OLD_ROOT.txt; files | sort -u; } > "$OUT.list"
    tar -cf "$OUT" -T "$OUT.list"
    rm -f OLD_ROOT.txt
    echo "wrote $OUT ($(du -h "$OUT" | cut -f1), $(wc -l < "$OUT.list") files; list in $OUT.list)" ;;
  *) echo "usage: $0 list|tar base|chain|full [OUT.tar] [--root DIR]"; exit 2 ;;
esac
