# Station Guardian PoC: synthetic CCTV for Story A (BTP Scenario 1)

Code that made the 90-second proof-of-concept video for Station Guardian: Story A (18 beats, A01–A18, 5 s each) and
its 35-second benign twin, played by two synthetic people on the **real camera view** of Gidea Park Platform 4
(camera GP02). One fixed camera, no audio, 16 fps, delivered at 1920×1080.

What the delivered videos contain and how they were checked: [docs/POC_REPORT.md](docs/POC_REPORT.md).

## How it works

```
real GP02 frame ──► plate (timestamp removed)          opening frame (woman seated on the bench)
                          │                                        │
                          ▼                                        ▼
   for each beat:  Wan2.2-I2V-A14B (+ LightX2V 4-step) image-to-video, 81 frames, from the previous beat's last frame
                   several seeds per beat ─► review in the fixed view ─► keep one take
                          │
                          ▼
   commit:  ground-plane registration to the plate (SIFT + RANSAC)  ─►  SAM3 people / instance masks
            ─►  composite people over the real plate with geometric cast shadows  ─►  per-garment colour correction
            ─►  corrected last frame seeds the next beat
                          │
                          ▼
   finish:  720p master ─► SeedVR2-7B upscale ─► re-lock people over ONE upscaled plate (1080p)
            ─► annotated / CCTV / clean videos + beat sheets
```

| Step | Model / method | Script |
|---|---|---|
| Beat generation | Wan2.2-I2V-A14B, fp16 experts + LightX2V 4-step adapters merged at load, first(+last)-frame mode | `scripts/wan_worker.py`, `scripts/submit.py` |
| Prompts and beat table | script text per beat + fixed scene / camera / people descriptions | `scripts/story.py` |
| Fixed camera | ground-plane homography to the plate | `scripts/bglock.py`, `scripts/lockjob.py` |
| Masks | SAM3 (text prompts: person, garments) | `scripts/sam3_*.py` |
| Composite + shadows | mattes, sun direction and shade measured on the plate | `scripts/compose.py` |
| Colour stability | per-garment Lab offsets | `scripts/colorfix.py` |
| Pinned beats | composed end frame + Wan2.2 first/last-frame mode | `scripts/end_from_take.py`, `pin_end.py`, `make_keyframe.py` |
| 1080p | SeedVR2-7B + background re-lock | `scripts/sr_stage.py` |
| Deliverables | videos, OSD clock, beat cards, beat sheets | `scripts/assemble.py`, `scripts/beatsheet.py` |

Step-by-step commands: [docs/PIPELINE.md](docs/PIPELINE.md).

## Quick start on a new server

Full guide: [docs/NEW_SERVER.md](docs/NEW_SERVER.md). In short:

```bash
git clone <this repo> station-guardian-poc && cd station-guardian-poc
setup/install_envs.sh all                 # conda envs "sg" and "sg-sam3", SeedVR2 clone + venv
cp config/env.example.sh config/env.sh    # set SG_PY, SG_SAM3_PY, SLURM account, data folder
source config/load_env.sh
setup/download_weights.sh                 # ~120 GB (Wan2.2, LightX2V, Z-Image-Turbo, SAM3, SeedVR2)
# copy the data that is not in git (plate, opening frame, kept takes) from the old server:
#   old server:  setup/data_bundle.sh tar chain sg_chain.tar
#   new server:  tar -xf sg_chain.tar -C "$SG_ROOT" && $SG_PY setup/relocate_paths.py --old "$(cat "$SG_ROOT/OLD_ROOT.txt")"
$SG_PY setup/check_setup.py               # everything OK?
slurm/submit.sh wan                       # GPU worker (or scripts/run_worker_local.sh without SLURM)
```

## Requirements

- Linux, NVIDIA GPU. The Wan2.2 worker holds both 14B experts in bf16 on one GPU: it was run on an H200 (141 GB),
  about 92 s per 81-frame take; plan on at least 80 GB of GPU memory and 160 GB host RAM.
- conda, git, ffmpeg (with libx264 and ffv1), DejaVu fonts.
- Hugging Face account with access to `facebook/sam3` (gated).
- Python environments: main (python 3.11, torch 2.9.1 cu128, diffusers 0.41, transformers 5.19), SAM3 (python 3.12),
  SeedVR2 venv on top of main. Pinned in `env/`; full package lists of the original server in `env/reference/`.

## Repository layout

```
scripts/     pipeline code (all paths come from scripts/sg_paths.py)
slurm/       GPU job files and submit.sh (account / partition / QOS from config/env.sh)
config/      env.example.sh (copy to env.sh), load_env.sh (sourced by every shell script)
setup/       install_envs.sh, download_weights.sh, check_setup.py, data_bundle.sh, relocate_paths.py
env/         requirements for the three environments; reference pip freezes
records/     chain.json of the delivered version: kept take, seed, prompt and lock stats per beat
docs/        POC_REPORT.md (deliverables, method, limits), PIPELINE.md, NEW_SERVER.md
```

Not in git (see `.gitignore`): model weights, generated takes and videos (`work/`, `chain/`, `output/`), the partner's
script documents and the real GP02 camera frame. The data folder (`SG_ROOT`, default: this folder) is moved between
servers with `setup/data_bundle.sh`.

## Hard limits of the content

Adults only; nothing sexual: no kiss, no groping, no hand on chest or under the coat, no nudity. Her response is a
two-handed shove, not a strike. No weapons, no injuries, no other people. Every video except the clean version carries
a SYNTHETIC tag. These limits are written into the prompts in `scripts/story.py` and were checked on every kept take.
