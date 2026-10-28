"""Check that a server is ready to run the pipeline. Runs on CPU in a few seconds; changes nothing.

  source config/load_env.sh && $SG_PY setup/check_setup.py

Prints one line per item: OK, MISSING (pipeline step that needs it) or WARN.
"""
import glob
import importlib
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import sg_paths as P  # noqa: E402

bad = 0


def report(ok, what, detail="", warn=False, hint=""):
    """detail: always shown; hint: shown only when the item is not OK."""
    global bad
    if not ok and hint:
        detail = (detail + "; " if detail else "") + hint
    tag = "OK     " if ok else ("WARN   " if warn else "MISSING")
    if not ok and not warn:
        bad += 1
    print(f"{tag} {what}" + (f"  ({detail})" if detail else ""))


def hf_file(repo, rel):
    hits = glob.glob(os.path.join(P.HF_HUB, "models--" + repo.replace("/", "--"), "snapshots", "*", rel))
    return bool(hits)


print(f"repo      {P.REPO}\ndata root {P.ROOT}\nHF hub    {P.HF_HUB}\n")

# 1. main environment
print(f"python    {sys.executable} ({sys.version.split()[0]})")
for mod in ["torch", "diffusers", "transformers", "accelerate", "safetensors", "cv2", "PIL", "numpy", "peft",
            "sentencepiece", "ftfy"]:
    try:
        m = importlib.import_module(mod)
        report(True, f"import {mod}", getattr(m, "__version__", ""))
    except Exception as e:  # noqa: BLE001
        report(False, f"import {mod}", f"{type(e).__name__}: {e}", hint="pip install -r env/requirements.txt")
try:
    import torch
    if torch.cuda.is_available():
        g = torch.cuda.get_device_properties(0)
        report(g.total_memory > 75e9, f"GPU {g.name} {g.total_memory / 1e9:.0f} GB",
               "the Wan2.2 worker was run on 141 GB; < 80 GB will not hold both experts", warn=True)
    else:
        report(False, "CUDA GPU", "none visible here (fine on a login node; the worker needs one)", warn=True)
except Exception:  # noqa: BLE001
    pass

# 2. system tools
for tool in ["ffmpeg", "ffprobe"]:
    report(shutil.which(tool) is not None, tool, hint="apt install ffmpeg / conda install -c conda-forge ffmpeg")
if shutil.which("ffmpeg"):
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    for e in ["libx264", "ffv1"]:
        report(f" {e} " in enc, f"ffmpeg encoder {e}")
for f in ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSansMono.ttf"]:
    report(os.path.exists(os.path.join(P.FONT_DIR, f)), f"font {f}", hint=f"not in SG_FONT_DIR={P.FONT_DIR}")

# 3. model files (setup/download_weights.sh)
W = [("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"),
     ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"),
     ("lightx2v/Wan2.2-Distill-Loras", "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors"),
     ("lightx2v/Wan2.2-Distill-Loras", "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"),
     ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "model_index.json"),
     ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "text_encoder/model.safetensors.index.json"),
     ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "vae/diffusion_pytorch_model.safetensors"),
     ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "transformer_2/config.json"),
     ("Tongyi-MAI/Z-Image-Turbo", "model_index.json"),
     ("Tongyi-MAI/Z-Image-Turbo", "transformer/diffusion_pytorch_model.safetensors.index.json")]
for repo, rel in W:
    report(hf_file(repo, rel), f"{repo}/{rel}", hint="setup/download_weights.sh")

# 4. SAM3 (separate environment; runs on the worker GPU)
report(os.path.exists(P.SAM3_CKPT), f"SAM3 checkpoint {P.SAM3_CKPT}", hint="setup/download_weights.sh sam3")
try:
    r = subprocess.run([P.SAM3_PY, "-c", "import sam3, sys; print(sys.version.split()[0])"], capture_output=True,
                       text=True, timeout=180)
    report(r.returncode == 0, f"SAM3 python {P.SAM3_PY} can import sam3",
           r.stdout.strip() if r.returncode == 0 else (r.stderr.strip().splitlines() or ["?"])[-1])
except Exception as e:  # noqa: BLE001
    report(False, f"SAM3 python {P.SAM3_PY}", f"{e}; set SG_SAM3_PY in config/env.sh")

# 5. SeedVR2 (1080p stage only)
report(os.path.exists(os.path.join(P.SEEDVR2_DIR, "inference_cli.py")), f"SeedVR2 code {P.SEEDVR2_DIR}",
       hint="setup/install_envs.sh seedvr2")
report(os.path.exists(P.SEEDVR2_PY), f"SeedVR2 python {P.SEEDVR2_PY}", hint="setup/install_envs.sh seedvr2")
for f in ["seedvr2_ema_7b_fp16.safetensors", "ema_vae_fp16.safetensors"]:
    report(os.path.exists(os.path.join(P.SEEDVR2_DIR, "models", "SEEDVR2", f)), f"SeedVR2 model {f}",
           hint="setup/download_weights.sh seedvr2")

# 6. data (copied from the original server with setup/data_bundle.sh; see docs/NEW_SERVER.md)
D = [("work/plates/plate.png", "base"), ("work/plates/plate_1536.png", "base"),
     ("work/plates/real/bench_front_mask.png", "base"), ("work/start/real/A01_start.png", "base"),
     ("work/colorfix/reference.json", "base"), ("chain/chain.json", "chain"), ("chain/A18_last_cc.png", "chain"),
     ("work/lock/A18/matte.npz", "chain"), ("work/plates/plate_sr", "chain")]
for rel, tier in D:
    report(os.path.exists(os.path.join(P.ROOT, rel)), f"data {rel}", hint=f"data bundle '{tier}'", warn=(tier != "base"))
if os.path.exists(os.path.join(P.ROOT, "chain", "chain.json")):
    import json
    c = json.load(open(os.path.join(P.ROOT, "chain", "chain.json")))
    stale = [b for b, r in c.get("beats", {}).items() if not str(r.get("clip", "")).startswith(P.ROOT)]
    report(not stale, "chain/chain.json paths point into the data root",
           hint=f"{len(stale)} beats point elsewhere: run setup/relocate_paths.py --old <old root>")

print("\nall required items present" if bad == 0 else f"\n{bad} required item(s) missing")
sys.exit(1 if bad else 0)
