#!/bin/bash
# Submit a GPU job with the account / partition / QOS from config/env.sh and logs in $SG_ROOT/logs.
#   slurm/submit.sh wan                      # Wan2.2 worker (default 4 h)
#   slurm/submit.sh wan --time=12:00:00      # extra sbatch arguments are passed through
#   PLATE_TAG=r1 slurm/submit.sh plate
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config/load_env.sh"
JOB="${1:-wan}"; shift || true
case "$JOB" in
  wan) SCRIPT="$SG_REPO/slurm/wan_worker.sbatch" ;;
  plate) SCRIPT="$SG_REPO/slurm/plate.sbatch" ;;
  *) echo "unknown job '$JOB' (wan | plate)" >&2; exit 2 ;;
esac
mkdir -p "$SG_ROOT/logs" "$SG_ROOT/work/queue"
ARGS=(--output="$SG_ROOT/logs/%x_%j.out" --export=ALL,SG_REPO="$SG_REPO")
[ -n "${SG_SLURM_ACCOUNT:-}" ] && ARGS+=(--account="$SG_SLURM_ACCOUNT")
[ -n "${SG_SLURM_PARTITION:-}" ] && ARGS+=(--partition="$SG_SLURM_PARTITION")
[ -n "${SG_SLURM_QOS:-}" ] && ARGS+=(--qos="$SG_SLURM_QOS")
sbatch "${ARGS[@]}" "$@" "$SCRIPT"
