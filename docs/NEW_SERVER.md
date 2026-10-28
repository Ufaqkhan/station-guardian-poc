# Setting up on a new server

About 1–2 hours, most of it downloads (~120 GB of models, plus 0.01–11 GB of data from the old server).

## 1. Code

```bash
git clone https://github.com/UfaqKhan/station-guardian-poc.git
cd station-guardian-poc
```

## 2. Python environments

```bash
setup/install_envs.sh all
```

This creates:

| Environment | Used for | Notes |
|---|---|---|
| conda `sg` (python 3.11) | everything except SAM3 | `env/requirements.txt`: torch 2.9.1 (CUDA 12.8), diffusers 0.41, transformers 5.19, opencv, peft |
| conda `sg-sam3` (python 3.12) | SAM3 masks (run on the worker GPU) | SAM3 from source, `third_party/sam3` at commit `5dd401d` |
| venv `tools/venv-seedvr2` | SeedVR2 1080p upscaling | on top of `sg` (`--system-site-packages`); code in `tools/seedvr2` at commit `4490bd1` |

Run the steps one by one with `setup/install_envs.sh main | sam3 | seedvr2`. Environment names can be changed with
`SG_ENV=... SG_SAM3_ENV=...`. Existing environments with the same packages work too: just point `config/env.sh` at them.
If a pinned version is not available for your CUDA driver, compare with `env/reference/pip-freeze-*.txt` (the full
package lists of the original server).

## 3. Settings

```bash
cp config/env.example.sh config/env.sh
```

Edit `config/env.sh`:

- `SG_PY`, `SG_SAM3_PY`: the python paths printed by `install_envs.sh`
  (e.g. `~/miniconda3/envs/sg/bin/python`, `~/miniconda3/envs/sg-sam3/bin/python`).
- `SG_ROOT`: where takes and videos go (needs ~60 GB for a full story). Default: the repository folder.
- `HF_HOME`: where the models are downloaded (~120 GB).
- `SG_SLURM_ACCOUNT`, `SG_SLURM_PARTITION`, `SG_SLURM_QOS`: your cluster's values (leave empty if not needed).

`config/env.sh` is git-ignored; every shell script loads it through `config/load_env.sh`, and the Python scripts read
the same variables (`scripts/sg_paths.py`).

## 4. Models

```bash
source config/load_env.sh
"$(dirname "$SG_PY")/hf" auth login   # Hugging Face token; needed for SAM3 (gated)
setup/download_weights.sh            # or one at a time: wan | zimage | sam3 | seedvr2
```

| Model | Hugging Face repo | Size |
|---|---|---|
| Wan2.2-I2V-A14B experts (fp16) | `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` (2 files) | 54 GB |
| LightX2V 4-step adapters | `lightx2v/Wan2.2-Distill-Loras` (2 files, `*_4step_1022`) | 1.3 GB |
| Wan2.2 text encoder, VAE, configs | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` (no transformer weights) | 12 GB |
| Z-Image-Turbo | `Tongyi-MAI/Z-Image-Turbo` | 31 GB |
| SAM3 | `facebook/sam3` → `sam3.pt` (accept the licence on the model page first) | 3.3 GB |
| SeedVR2-7B + VAE | `numz/SeedVR2_comfyUI` (`seedvr2_ema_7b_fp16`, `ema_vae_fp16`) | 16 GB |

## 5. Data from the old server

The real camera plate, the opening frame and the generated takes are not in git. On the **old** server (in its copy
of this repository, with its `config/env.sh` pointing `SG_ROOT` at the old data folder, or with `--root`):

```bash
setup/data_bundle.sh tar base  sg_base.tar  --root /share_2/users/umair_nawaz/World-Models/SANA-Work/PoC   # ~10 MB
setup/data_bundle.sh tar chain sg_chain.tar --root /share_2/users/umair_nawaz/World-Models/SANA-Work/PoC  # ~4.3 GB
setup/data_bundle.sh tar full  sg_full.tar  --root /share_2/users/umair_nawaz/World-Models/SANA-Work/PoC   # ~11.5 GB
```

- `base`: plate, bench mask, opening frame, colour reference, `chain.json` → enough to make new beats or a new story
  on the GP02 view.
- `chain`: + all kept beats, their raw takes, masks, mattes, colour offsets → re-compose / re-render Story A or
  continue from any beat.
- `full`: + 720p/1080p masters and the delivered videos.

Copy the tar (`scp`, `rsync`) and on the **new** server:

```bash
source config/load_env.sh
mkdir -p "$SG_ROOT" && tar -xf sg_chain.tar -C "$SG_ROOT"
$SG_PY setup/relocate_paths.py --old "$(cat "$SG_ROOT/OLD_ROOT.txt")"     # rewrite stored absolute paths
```

Without the old server, `records/chain.json` still documents the kept take, seed and prompt of every beat.

## 6. Check

```bash
$SG_PY setup/check_setup.py
```

Every line should read OK (GPU and data lines may read WARN on a login node or before step 5).

## 7. Run

```bash
slurm/submit.sh wan           # or: GPU=0 scripts/run_worker_local.sh
```

Then follow [PIPELINE.md](PIPELINE.md). A first smoke test that touches every model: queue one take from the
opening frame and commit it:

```bash
cd "$SG_ROOT"
$SG_PY $SG_REPO/scripts/submit.py --beat A01 --start work/start/real/A01_start.png --seeds 1 --tag smoke
# wait for work/beats/A01/smoke/A01_smoke_s1.mkv (about 2 min after the model has loaded), then look at it
```

(Committing that take with `scripts/commit.sh` would overwrite `chain/A01.*`; do it only if you want to replace the
beat.)
