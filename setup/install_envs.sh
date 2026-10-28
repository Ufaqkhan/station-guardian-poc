#!/bin/bash
# Create the three Python environments the pipeline uses.
#   setup/install_envs.sh main      # conda env "sg"       (python 3.11): torch 2.9.1 cu128, diffusers 0.41, ...
#   setup/install_envs.sh sam3      # conda env "sg-sam3"  (python 3.12) + SAM3 from source (third_party/sam3)
#   setup/install_envs.sh seedvr2   # tools/seedvr2 (pinned clone) + tools/venv-seedvr2 on top of "sg"
#   setup/install_envs.sh all
# Needs conda (or mamba) on PATH, git, and an NVIDIA driver for CUDA 12.8. Env names: SG_ENV / SG_SAM3_ENV.
# Afterwards put the printed python paths into config/env.sh (SG_PY, SG_SAM3_PY).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SG_ENV="${SG_ENV:-sg}"
SG_SAM3_ENV="${SG_SAM3_ENV:-sg-sam3}"
SAM3_COMMIT=5dd401d1c5c1d5c3eedff06d41b77af824517619
SEEDVR2_COMMIT=4490bd1f482e026674543386bb2a4d176da245b9
CONDA="$(command -v mamba || command -v conda || true)"

env_python() { "$CONDA" run -n "$1" python -c "import sys; print(sys.executable)"; }

main_env() {
  [ -n "$CONDA" ] || { echo "conda not found"; exit 1; }
  "$CONDA" create -y -n "$SG_ENV" python=3.11
  "$CONDA" run -n "$SG_ENV" python -m pip install -r "$REPO/env/requirements.txt"
  echo "SG_PY=$(env_python "$SG_ENV")"
}

sam3_env() {
  [ -n "$CONDA" ] || { echo "conda not found"; exit 1; }
  mkdir -p "$REPO/third_party"
  if [ ! -d "$REPO/third_party/sam3/.git" ]; then
    git clone https://github.com/facebookresearch/sam3 "$REPO/third_party/sam3"
  fi
  git -C "$REPO/third_party/sam3" checkout "$SAM3_COMMIT"
  "$CONDA" create -y -n "$SG_SAM3_ENV" python=3.12
  "$CONDA" run -n "$SG_SAM3_ENV" python -m pip install -r "$REPO/env/requirements-sam3.txt"
  "$CONDA" run -n "$SG_SAM3_ENV" python -m pip install -e "$REPO/third_party/sam3"
  echo "SG_SAM3_PY=$(env_python "$SG_SAM3_ENV")"
}

seedvr2() {
  local PY
  PY="${SG_PY:-$(env_python "$SG_ENV")}"
  mkdir -p "$REPO/tools"
  if [ ! -d "$REPO/tools/seedvr2/.git" ]; then
    git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler "$REPO/tools/seedvr2"
  fi
  git -C "$REPO/tools/seedvr2" checkout "$SEEDVR2_COMMIT"
  "$PY" -m venv --system-site-packages "$REPO/tools/venv-seedvr2"
  "$REPO/tools/venv-seedvr2/bin/python" -m pip install -r "$REPO/env/requirements-seedvr2.txt"
  echo "SG_SEEDVR2_DIR=$REPO/tools/seedvr2  SG_SEEDVR2_PY=$REPO/tools/venv-seedvr2/bin/python"
}

case "${1:-all}" in
  main) main_env ;;
  sam3) sam3_env ;;
  seedvr2) seedvr2 ;;
  all) main_env; sam3_env; seedvr2 ;;
  *) echo "usage: $0 main|sam3|seedvr2|all"; exit 2 ;;
esac
