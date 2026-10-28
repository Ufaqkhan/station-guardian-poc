"""Server-specific paths, in one place.

Every value comes from an environment variable when it is set (see config/env.example.sh); otherwise the default
below is used. Nothing else in scripts/ should contain an absolute path.

  REPO        this repository (code)
  ROOT        data folder: work/, chain/, output/, logs/  (SG_ROOT, default = REPO)
  SAM3_PY     python of the SAM3 environment               (SG_SAM3_PY)
  SAM3_CKPT   SAM3 checkpoint sam3.pt                       (SG_SAM3_CKPT)
  HF_HUB      Hugging Face hub cache with the Wan2.2 files  (HF_HUB_CACHE, or HF_HOME/hub)
  SEEDVR2_DIR clone of numz/ComfyUI-SeedVR2_VideoUpscaler   (SG_SEEDVR2_DIR)
  SEEDVR2_PY  python of the SeedVR2 venv                    (SG_SEEDVR2_PY)
  FONT_DIR    folder with DejaVuSans*.ttf                   (SG_FONT_DIR)
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")
ROOT = os.path.abspath(os.path.expanduser(os.environ.get("SG_ROOT") or REPO))
SAM3_PY = os.environ.get("SG_SAM3_PY") or "python"
SAM3_CKPT = os.environ.get("SG_SAM3_CKPT") or os.path.join(REPO, "weights", "sam3", "sam3.pt")
HF_HUB = os.environ.get("HF_HUB_CACHE") or os.path.join(
    os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
SEEDVR2_DIR = os.environ.get("SG_SEEDVR2_DIR") or os.path.join(REPO, "tools", "seedvr2")
SEEDVR2_PY = os.environ.get("SG_SEEDVR2_PY") or os.path.join(REPO, "tools", "venv-seedvr2", "bin", "python")
FONT_DIR = os.environ.get("SG_FONT_DIR") or "/usr/share/fonts/truetype/dejavu"
