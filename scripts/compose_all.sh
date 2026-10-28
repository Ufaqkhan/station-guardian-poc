#!/bin/bash
# Re-compose every committed beat (waits for each beat's SAM3 instances), then write the 720p masters.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config/load_env.sh"
cd "$SG_ROOT"
export CUDA_VISIBLE_DEVICES=""
for b in A01 A02 A03 A04 A05 A06 A07 A08 A09 A10 A11 A12 A13 A14 A15 A16 A17 A18 AT1 AT2 AT3 AT4; do
  until [ -f "work/lock/$b/instances.npz" ]; do sleep 10; done
  "$SG_PY" "$SG_REPO/scripts/compose.py" "$b" 2>&1 | grep -E "composed|Traceback|Error"
done
echo COMPOSE_ALL_DONE
"$SG_PY" "$SG_REPO/scripts/assemble.py" --write_master work/master/story_720.mkv 2>&1 | grep -E "master:|Traceback"
"$SG_PY" "$SG_REPO/scripts/assemble.py" --twin --write_master work/master/twin_720.mkv 2>&1 | grep -E "master:|Traceback"
echo MASTERS_DONE
