#!/bin/bash
# Persistent Wan2.2-I2V-A14B worker: loads both 14B experts once, then serves requests from $SG_ROOT/work/queue
# (generation takes, plus "cmd" requests for SAM3 and SeedVR2) until idle for IDLE_MIN minutes or work/queue/STOP exists.
# Submit with:  slurm/submit.sh wan [extra sbatch args, e.g. --time=12:00:00]
# Tested on 1x H200 (141 GB). Needs ~160 GB host RAM while the experts are merged and moved to the GPU.
#SBATCH --job-name=sg_wan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${SG_REPO:?submit with slurm/submit.sh so SG_REPO is passed to the job}"
source "$SG_REPO/config/load_env.sh"
cd "$SG_ROOT"
mkdir -p work/queue
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
echo "host $(hostname) job ${SLURM_JOB_ID:-local} start $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$SG_PY" "$SG_REPO/scripts/wan_worker.py" --queue "$SG_ROOT/work/queue" --idle_min "${IDLE_MIN:-25}" \
  --max_hours "${MAX_HOURS:-3.75}" ${WORKER_ARGS:-}
echo "end $(date -Is)"
