#!/bin/bash
# Download the models the pipeline uses (about 90 GB; +31 GB with the optional Z-Image-Turbo). Uses the `hf` CLI from huggingface_hub
# (in the main environment). Files go to the Hugging Face cache ($HF_HOME/hub) except SAM3 and SeedVR2.
#   setup/download_weights.sh            # all models used by the delivered version (wan, sam3, seedvr2)
#   setup/download_weights.sh wan        # only one: wan | sam3 | seedvr2 | zimage (optional, first-version tools)
# SAM3 is gated: accept the licence on https://huggingface.co/facebook/sam3 and run `hf auth login` first.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config/load_env.sh"
HFCLI="$(dirname "$SG_PY")/hf"
[ -x "$HFCLI" ] || HFCLI="$(command -v hf)"
what="${1:-all}"

if [[ $what == all || $what == wan ]]; then
  # Wan2.2-I2V-A14B experts, fp16 single files (ComfyUI repackage), ~2 x 28.6 GB
  "$HFCLI" download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    --include "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
              "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
  # LightX2V 4-step distillation adapters (merged into the experts at load time)
  "$HFCLI" download lightx2v/Wan2.2-Distill-Loras \
    --include "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors" \
              "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
  # Diffusers pipeline parts: configs, UMT5 text encoder, tokenizer, VAE (NOT the transformer weights)
  "$HFCLI" download Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    --include "model_index.json" "scheduler/*" "text_encoder/*" "tokenizer/*" "vae/*" \
              "transformer/config.json" "transformer_2/config.json"
fi
if [[ $what == zimage ]]; then
  # Z-Image-Turbo (optional): only for the first-version tools make_start.py, clean_osd.py, make_plate.py
  "$HFCLI" download Tongyi-MAI/Z-Image-Turbo --exclude "assets/*"
fi
if [[ $what == all || $what == sam3 ]]; then
  mkdir -p "$(dirname "$SG_SAM3_CKPT")"
  "$HFCLI" download facebook/sam3 sam3.pt --local-dir "$(dirname "$SG_SAM3_CKPT")"
fi
if [[ $what == all || $what == seedvr2 ]]; then
  mkdir -p "$SG_SEEDVR2_DIR/models/SEEDVR2"
  "$HFCLI" download numz/SeedVR2_comfyUI seedvr2_ema_7b_fp16.safetensors ema_vae_fp16.safetensors \
    --local-dir "$SG_SEEDVR2_DIR/models/SEEDVR2"
fi
echo "done; check with: \$SG_PY setup/check_setup.py"
