#!/bin/bash
# Run the Wan2.2 worker without SLURM, on one local GPU, in the background. Log: $SG_ROOT/logs/worker_<time>.out
#   GPU=0 scripts/run_worker_local.sh            # stop it with:  touch "$SG_ROOT/work/queue/STOP"
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config/load_env.sh"
mkdir -p "$SG_ROOT/logs" "$SG_ROOT/work/queue"
rm -f "$SG_ROOT/work/queue/STOP"
LOG="$SG_ROOT/logs/worker_$(date +%Y%m%d_%H%M%S).out"
CUDA_VISIBLE_DEVICES="${GPU:-0}" SG_REPO="$SG_REPO" MAX_HOURS="${MAX_HOURS:-48}" \
  nohup bash "$SG_REPO/slurm/wan_worker.sbatch" > "$LOG" 2>&1 &
echo "worker pid $! log $LOG"
