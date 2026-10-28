#!/bin/bash
# Candidate stills of an empty platform view with Z-Image-Turbo (only needed for a NEW camera view without a real frame).
# Submit with:  PLATE_TAG=r1 slurm/submit.sh plate
#SBATCH --job-name=sg_plate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH --no-requeue

set -euo pipefail
: "${SG_REPO:?submit with slurm/submit.sh so SG_REPO is passed to the job}"
source "$SG_REPO/config/load_env.sh"
cd "$SG_ROOT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
echo "host $(hostname) job ${SLURM_JOB_ID:-local} start $(date -Is)"
"$SG_PY" "$SG_REPO/scripts/make_plate.py" --out "$SG_ROOT/work/plates/${PLATE_TAG:-r1}" ${PLATE_ARGS:-}
echo "end $(date -Is)"
