#!/bin/bash
# Commit one take to the chain:
#   SAM3 people masks + ground homographies (lockjob) -> SAM3 instances -> compose (matte + geometric shadows)
#   -> garment colour measurement -> colour-corrected last frame (start image for the next beat)
# Needs a running worker (slurm/submit.sh wan or scripts/run_worker_local.sh): SAM3 runs on the worker GPU.
# usage: scripts/commit.sh A16 work/beats/A16/k3/A16_k3_s5.mkv      (take path relative to $SG_ROOT, or absolute)
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config/load_env.sh"
cd "$SG_ROOT"
B=$1; T=$(realpath "$2")
export CUDA_VISIBLE_DEVICES=""
"$SG_PY" "$SG_REPO/scripts/lockjob.py" "$B" "$T" 2>&1 | grep -E "^locked|aligned|Traceback|Error" || true
rm -f "work/lock/$B/instances.npz"
N=inst_${B}_$(date +%s)
printf '{"name": "%s", "cmd": "%s %s/scripts/sam3_instances.py --video %s --out %s/work/lock/%s/instances.npz", "priority": -1}' \
  "$N" "$SG_SAM3_PY" "$SG_REPO" "$T" "$SG_ROOT" "$B" > "work/queue/.$N.tmp" && mv "work/queue/.$N.tmp" "work/queue/$N.json"
until [ -f "work/lock/$B/instances.npz" ]; do
  sleep 5
  [ -n "$(ls work/queue/failed/*$N* 2>/dev/null)" ] && { echo "instances failed"; exit 1; }
done
"$SG_PY" "$SG_REPO/scripts/compose.py" "$B" 2>&1 | grep -E "composed|Traceback|Error"
"$SG_PY" "$SG_REPO/scripts/colorfix.py" seed "$B" 2>&1 | grep -E "wrote|Traceback|Error"
echo "COMMITTED $B $T"
