# Station Guardian PoC: server settings.
# Copy to config/env.sh (git-ignored) and edit for your server:   cp config/env.example.sh config/env.sh
# Every shell script sources this through config/load_env.sh; Python reads the same variables (scripts/sg_paths.py).
# $SG_REPO (this repository) is set before this file is read.

# Data folder: work/, chain/, output/, logs/ are created here. Default: the repository itself (all git-ignored).
export SG_ROOT="${SG_ROOT:-$SG_REPO}"

# Python of the main environment (torch + diffusers; setup/install_envs.sh main). Example: ~/miniconda3/envs/sg/bin/python
export SG_PY="${SG_PY:-$HOME/miniconda3/envs/sg/bin/python}"

# SAM3: its own environment (python 3.12) and checkpoint (setup/install_envs.sh sam3; setup/download_weights.sh sam3)
export SG_SAM3_PY="${SG_SAM3_PY:-$HOME/miniconda3/envs/sg-sam3/bin/python}"
export SG_SAM3_CKPT="${SG_SAM3_CKPT:-$SG_REPO/weights/sam3/sam3.pt}"

# SeedVR2 (1080p upscale): code clone + venv (setup/install_envs.sh seedvr2)
export SG_SEEDVR2_DIR="${SG_SEEDVR2_DIR:-$SG_REPO/tools/seedvr2}"
export SG_SEEDVR2_PY="${SG_SEEDVR2_PY:-$SG_REPO/tools/venv-seedvr2/bin/python}"

# Hugging Face cache (Wan2.2, LightX2V, Z-Image-Turbo are read from $HF_HOME/hub)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Fonts for the annotated video (DejaVuSans.ttf, DejaVuSans-Bold.ttf). Debian/Ubuntu: apt install fonts-dejavu-core
export SG_FONT_DIR="${SG_FONT_DIR:-/usr/share/fonts/truetype/dejavu}"

# SLURM (only used by slurm/submit.sh). Leave SG_SLURM_ACCOUNT empty if your cluster has no accounts.
export SG_SLURM_ACCOUNT="${SG_SLURM_ACCOUNT:-}"
export SG_SLURM_PARTITION="${SG_SLURM_PARTITION:-}"
export SG_SLURM_QOS="${SG_SLURM_QOS:-}"
