# Station Guardian PoC: synthetic CCTV for Story A (BTP Scenario 1)

This repository holds the code that made the proof-of-concept video for Station Guardian. The video plays Story A of the
partner script (*"The story, beat by beat"*, 18 beats A01–A18, 5 s each) and its benign twin (A01–A03 + AT1–AT4).
Two synthetic people act on the **real camera view** of Gidea Park Platform 4, camera GP02. There is one fixed camera,
no audio, 16 fps, and the videos are delivered at 1920×1080.

This README covers:

1. [What was produced](#1-what-was-produced)
2. [What was run to produce it](#2-what-was-run-to-produce-it): environment, every step, and every beat
3. [How to generate it yourself](#3-how-to-generate-it-yourself): reproduce, re-render, or make new beats
4. [Checks and known limitations](#4-checks-and-known-limitations)
5. [Repository layout and script reference](#5-repository-layout-and-script-reference)

More detail: [docs/NEW_SERVER.md](docs/NEW_SERVER.md) (setup), [docs/PIPELINE.md](docs/PIPELINE.md) (commands and
lessons), [docs/POC_REPORT.md](docs/POC_REPORT.md) (the report delivered with the videos).

---

## 1. What was produced

Delivered on 2026-10-22. The files are not in git (see [section 3.2](#32-get-the-data-that-is-not-in-git)).

| File (in `output/`) | Content | Size |
|---|---|---|
| `StoryA_GP02_annotated.mp4` | **Main PoC video.** CCTV view plus a beat card (beat id, time, storyboard panel, the script's "what the camera sees" text, behaviour label, tier) and a tier timeline | 1920×1080, 1440 frames, 90.0 s |
| `StoryA_GP02_cctv.mp4` | Camera view with the GP02 camera's own on-screen clock (recreated) and a SYNTHETIC tag | 1920×1080, 90.0 s |
| `StoryA_GP02_clean.mp4` | Camera picture only, no overlays | 1920×1080, 90.0 s |
| `StoryA_GP02_beat_sheet.jpg` | One page: the middle frame of each beat with id, time, behaviour and tier | |
| `StoryA_twin_GP02_{annotated,cctv,clean}.mp4`, `StoryA_twin_GP02_beat_sheet.jpg` | The same for the benign twin | 560 frames, 35.0 s |

Each beat is 80 frames, which is 5.0 s at 16 fps.

---

## 2. What was run to produce it

### 2.1 Hardware and software

| | |
|---|---|
| GPU | 1× NVIDIA H200 (141 GB), SLURM (`debug` QOS, 1 GPU, 4 h jobs), one persistent worker job at a time |
| Generation speed | 81-frame take at 1280×720: **median 91 s** (90–98 s) after the model has loaded (~4 min) |
| Main environment | python 3.11, torch 2.9.1 + CUDA 12.8, diffusers 0.41.0, transformers 5.19.0, opencv 5.0, peft 0.21.2 (`env/requirements.txt`) |
| SAM3 environment | python 3.12, SAM3 from github.com/facebookresearch/sam3 at commit `5dd401d` (`env/requirements-sam3.txt`) |
| SeedVR2 | github.com/numz/ComfyUI-SeedVR2_VideoUpscaler at commit `4490bd1`, venv on top of the main environment |
| System | ffmpeg 6.1 (libx264, ffv1), DejaVu fonts |

Full package lists of the original server are in `env/reference/`.

### 2.2 Models

| Model | Used for | Settings used |
|---|---|---|
| **Wan2.2-I2V-A14B** (Comfy-Org fp16 high- and low-noise experts) + **LightX2V 4-step adapters** (`*_lightx2v_4step_1022`) | every beat take | adapters merged into the weights at load time; 4 steps, sigmas [1.0, 0.75, 0.5, 0.25], FlowMatch shift 5; guidance 1.0; 81 frames, 1280×720, 16 fps; first-frame mode, or first + last frame mode for pinned beats; default negative prompt in `wan_worker.py` |
| **SAM3** (`facebook/sam3`, `sam3.pt`) | person masks, per-person instances, garment masks | text prompts ("person", "handbag", garment names, "woman in a red coat sitting on a bench", ...) |
| **SeedVR2-7B** (`seedvr2_ema_7b_fp16` + `ema_vae_fp16`) | 720p → 1080p | batch 25, temporal overlap 4, colour correction "lab" (`sr_stage.py`) |
| Z-Image-Turbo | tried for timestamp removal and a stand-in platform (v1); **not used in the delivered version** | |

Why Wan2.2-I2V-A14B: it was the strongest open image-to-video model for human motion and for following action
descriptions. The 4-step distillation made 6 takes per beat affordable.

### 2.3 Steps, in the order they were run

All commands are run from the data folder (`cd $SG_ROOT`) with `source config/load_env.sh` done first.
`S=$SG_REPO/scripts`.

**Step 1: plate (empty platform).** The plate is the real GP02 frame (`View-Display.jpeg`, 1600×900) with the camera's
burned-in timestamp removed. Inside the translucent box the scene is still visible, measured as
observed = 0.318 × scene + 10.1. The script inverts that, inpaints the text strokes and matches the noise to the rows
below.
```bash
$SG_PY $S/make_real_plate.py --src View-Display.jpeg
# -> work/plates/plate.png (1280x720, used everywhere), work/plates/plate_1536.png (1600x900), work/plates/real/*
```

**Step 2: opening frame (A01 start).** First, Wan2.2 generated the woman walking in and sitting on the bench
(`P0a` prompt in `story.py`, 3 seeds; seed 3 kept). Then frame 80 was cut out with SAM3 and moved along the bench. The
person is scaled by the measured perspective law h = 0.75·(foot_y + 36), and the bench's front rail and near armrest
stay in front of her (`work/plates/real/bench_front_mask.png`, a hand-made mask).
```bash
$SG_PY $S/submit.py --beat P0a --tag g1 --start work/plates/plate.png --seeds 1 2 3       # needs the worker (step 3)
$SG_PY $S/take_frame.py work/beats/P0a/g1/P0a_g1_s3.mkv --idx 80 --out work/start/real/P0a_g1_s3_f80.png \
    --mask "woman in a red coat sitting on a bench"
$SG_PY $S/move_seated.py --src work/start/real/P0a_g1_s3_f80.png --mask work/start/real/P0a_g1_s3_f80_mask.png \
    --frac 0.20 --out work/start/real/A01_start.png
$SG_PY $S/chain.py init --plate work/plates/plate.png --start work/start/real/A01_start.png
```

**Step 3: GPU worker.** One job loads both 14B experts once. It then serves the queue folder `work/queue/`: takes,
plus SAM3 and SeedVR2 "cmd" requests (SAM3 needs CUDA, so it runs on the worker too).
```bash
slurm/submit.sh wan          # or, without SLURM: GPU=0 scripts/run_worker_local.sh
```

**Step 4: beats, one after another.** Each beat starts from the previous beat's colour-corrected last frame. For each
beat:
1. Queue 6 seeds (4 for the twin).
2. Review them in the final fixed view, checking the script line, the hard limits, camera stability, pose and size.
3. Commit the best one.

```bash
$SG_PY $S/submit.py --beat A06 --tag g1 --start chain/A05_last_cc.png --seeds 1 2 3 4 5 6 [--prompt "..."]
$SG_PY $S/rev.py A06 g1 --n 6                         # registered review grid + camera movement per take
scripts/commit.sh A06 work/beats/A06/g1/A06_g1_s1.mkv  # register + SAM3 + composite + shadows + colour + next seed
```
`commit.sh` runs four steps:
1. `lockjob.py`: ground-plane homography of every frame to the plate (SIFT + RANSAC on the platform surface) and SAM3
   person masks.
2. `sam3_instances.py`: per-person masks.
3. `compose.py`: the people are cut out with 1 px mattes and placed over the real plate. Each person gets a geometric
   cast shadow along the sun direction measured on the plate. No shadow is added where the plate is already in shade.
4. `colorfix.py seed`: per-garment Lab correction towards the colours of A01/A05, which writes
   `chain/<beat>_last_cc.png` for the next beat.

**Pinned beats** (A02, A03, A04, A10, A13, A14, AT1) were used where every take zoomed after the people or drifted into a
wrong pose (for example the man reclining or falling off the bench). For these, the end frame was composed in the fixed
view: the plate plus people cut from a registered frame of an earlier take (`take_frame.py --reg`, then
`end_from_take.py` or `pin_end.py`). Wan2.2 then generated the motion between the two frames (`submit.py --last`).

Many kept takes use a prompt tuned for that beat (`--prompt`) instead of the `story.py` default. The exact request of
every kept take is saved in `records/kept_takes.json`, and `scripts/replay.py` re-queues it.

**Step 5: 720p masters.** The committed beats are joined, with an 8-frame ease at every boundary into the next beat's
first frame.
```bash
scripts/compose_all.sh       # -> work/master/story_720.mkv, work/master/twin_720.mkv (lossless)
```

**Step 6: 1080p.** SeedVR2-7B upscales the masters and the plate on the worker. The upscaled people are then put back
over **one** upscaled plate with the same mattes and shadows, so the background does not shimmer.
```bash
$SG_PY $S/sr_stage.py queue --master work/master/story_720.mkv --tag story
$SG_PY $S/sr_stage.py queue --master work/master/twin_720.mkv --tag twin
```

**Step 7: deliverables.** This step re-locks at 1080p, builds the three videos per story (recreated GP02 clock from
13:36:34, SYNTHETIC tag, beat cards), and writes the beat sheets.
```bash
scripts/render_final.sh      # -> output/
```

### 2.4 Every beat as run

The takes column counts the takes generated on the real view for that beat, across all rounds. Round `g` = free
generation, round `k` = pinned end frame. Seeds, prompts and frames are in `records/kept_takes.json`.

| Beat | Time | Behaviour | Tier | Takes | Kept take | Mode |
|---|---|---|---|---|---|---|
| A01 | 00–05 s | waiting stationary | T0 | 6 | `A01_g1_s1` (seed 1) | first frame |
| A02 | 05–10 s | waiting stationary | T0 | 12 | `A02_k1_s3` (seed 3) | first + last frame |
| A03 | 10–15 s | walking through | T0 | 12 | `A03_k1_s1` (seed 1) | first + last frame |
| A04 | 15–20 s | personal space encroachment | T1 | 24 | `A04_k1_s3` (seed 3) | first + last frame |
| A05 | 20–25 s | unwanted proximity seated | T2 | 12 | `A05_g1_s2` (seed 2) | first frame |
| A06 | 25–30 s | low level harassment gesture | T2 | 6 | `A06_g1_s1` (seed 1) | first frame |
| A07 | 30–35 s | low level harassment gesture | T2 | 6 | `A07_g1_s3` (seed 3) | first frame |
| A08 | 35–40 s | low level harassment gesture | T2 | 6 | `A08_g1_s3` (seed 3) | first frame |
| A09 | 40–45 s | low level harassment gesture | T2 | 6 | `A09_g1_s4` (seed 4) | first frame |
| A10 | 45–50 s | unwanted proximity seated | T2 | 18 | `A10_k1_s2` (seed 2) | first + last frame |
| A11 | 50–55 s | unwanted touching | T3 | 24 | `A11_g4_s1` (seed 1) | first frame |
| A12 | 55–60 s | unwanted touching | T3 | 6 | `A12_g1_s3` (seed 3) | first frame |
| A13 | 60–65 s | unwanted touching | T3 | 12 | `A13_k1_s1` (seed 1) | first + last frame |
| A14 | 65–70 s | grab or assault attempt | T4 | 12 | `A14_k1_s4` (seed 4) | first + last frame |
| A15 | 70–75 s | grab or assault attempt | T4 | 12 | `A15_g2_s10` (seed 10) | first frame |
| A16 | 75–80 s | grab or assault attempt | T4 | 12 | `A16_g1_s1` (seed 1) | first frame |
| A17 | 80–85 s | confrontational posture | T3 | 24 | `A17_g1_s3` (seed 3) | first frame |
| A18 | 85–90 s | confrontational posture | T3 | 18 | `A18_g1_s3` (seed 3) | first frame |
| AT1 | twin 15–20 s | waiting stationary | T0 | 8 | `AT1_k1_s1` (seed 1) | first + last frame |
| AT2 | twin 20–25 s | walking through | T0 | 8 | `AT2_g1_s2` (seed 2) | first frame |
| AT3 | twin 25–30 s | walking through | T0 | 4 | `AT3_g1_s3` (seed 3) | first frame |
| AT4 | twin 30–35 s | walking through | T0 | 4 | `AT4_g1_s3` (seed 3) | first frame |

In total there were **252 beat takes plus 12 walk-in takes**, about 7 GPU-hours of generation. The twin shares A01–A03
with the story and branches from A03's last frame.

Beats that changed the script slightly:
- A15: she holds both palms up after the shove.
- A17: she backs away about 4–5 m.
- The evening light is not done (afternoon sun only).

See [docs/POC_REPORT.md](docs/POC_REPORT.md).

---

## 3. How to generate it yourself

### 3.1 Set up a server

```bash
git clone https://github.com/Ufaqkhan/station-guardian-poc.git && cd station-guardian-poc
setup/install_envs.sh all                 # conda envs "sg" + "sg-sam3", SeedVR2 clone + venv
cp config/env.example.sh config/env.sh    # set SG_PY, SG_SAM3_PY, SG_ROOT, HF_HOME, SLURM account/partition/QOS
source config/load_env.sh
"$(dirname "$SG_PY")/hf" auth login       # SAM3 is gated: accept its licence on huggingface.co/facebook/sam3 first
setup/download_weights.sh                 # ~90 GB
$SG_PY setup/check_setup.py               # every line should say OK
```
Details and troubleshooting: [docs/NEW_SERVER.md](docs/NEW_SERVER.md). You need a GPU with at least 80 GB (141 GB
used) and 160 GB host RAM for the worker.

### 3.2 Get the data that is not in git

The real camera frame, the plate, the opening frame, the pinned end frames and the generated takes stay out of git.
Pack them on the original server and unpack them on the new one:

```bash
# original server (code copy in SANA-Work/station-guardian-poc, data in SANA-Work/PoC)
setup/data_bundle.sh tar chain sg_chain.tar --root /share_2/users/umair_nawaz/World-Models/SANA-Work/PoC
# new server
tar -xf sg_chain.tar -C "$SG_ROOT" && $SG_PY setup/relocate_paths.py --old "$(cat "$SG_ROOT/OLD_ROOT.txt")"
```

| Bundle | Size | Contains | Lets you |
|---|---|---|---|
| `base` | ~10 MB | plate (720p, full size, SeedVR2-upscaled), bench mask, opening frame, colour reference, chain.json | make new beats or a new story on GP02 |
| `chain` | ~4.3 GB | + every kept beat, its raw take, masks, homographies, mattes, colour offsets, pinned end frames | replay, re-compose and re-render Story A, or continue from any beat |
| `full` | ~11.5 GB | + 720p/1080p masters and the delivered videos | redo only the final assembly |

`View-Display.jpeg` (needed only for step 1) is partner material. Copy it by hand if you need to rebuild the plate.

### 3.3 Choose what to do

**A. Re-render the delivered videos** (`full` bundle, CPU only):
```bash
cd "$SG_ROOT" && $SG_REPO/scripts/render_final.sh
```

**B. Re-compose from the kept takes** (`chain` bundle; for example after changing shadows in `compose.py` or the
colour correction in `colorfix.py`). The worker must be running, because the colour measurement uses SAM3:
```bash
cd "$SG_ROOT"; S=$SG_REPO/scripts
for b in A01 A02 A03 A04 A05 A06 A07 A08 A09 A10 A11 A12 A13 A14 A15 A16 A17 A18 AT1 AT2 AT3 AT4; do
  $SG_PY $S/compose.py $b && $SG_PY $S/colorfix.py measure $b; done
$SG_PY $S/assemble.py --write_master work/master/story_720.mkv            # 720p masters (colour-corrected)
$SG_PY $S/assemble.py --twin --write_master work/master/twin_720.mkv
$SG_PY $S/sr_stage.py queue --master work/master/story_720.mkv --tag story  # SeedVR2 on the worker
$SG_PY $S/sr_stage.py queue --master work/master/twin_720.mkv --tag twin
$SG_REPO/scripts/render_final.sh
```

**C. Regenerate beats** (worker running):
```bash
$SG_PY $SG_REPO/scripts/replay.py --list                    # what each kept take was made from
$SG_PY $SG_REPO/scripts/replay.py A06                       # exact request -> work/beats/A06/replay/
$SG_PY $SG_REPO/scripts/replay.py A06 --seeds 1 2 3 4 5 6   # same request, new seeds to choose from
$SG_PY $SG_REPO/scripts/rev.py A06 replay --n 6             # review, then:
$SG_REPO/scripts/commit.sh A06 work/beats/A06/replay/A06_replay_s3.mkv
```
Committing a beat changes its last frame. The beats after it were generated from the old frame, so either regenerate
them too, or ease the new beat into the old next beat with `blend_tail.py`. GPU kernels are not bit-exact across
machines, so replays are close to the kept takes but not identical.

**D. Make a new story from scratch** (`base` bundle, or a new camera):
1. Edit `scripts/story.py`: the beat table (`BEATS`, `TWIN`), the scene, camera and people descriptions (`CAMERA`,
   `WOMAN`, `MAN`), and the prompt per beat.
2. For a new camera view, run steps 1–2 of [section 2.3](#23-steps-in-the-order-they-were-run) on its frame, and
   re-measure the view constants listed in [docs/PIPELINE.md](docs/PIPELINE.md#constants-measured-on-the-gp02-view):
   shadow direction and length, perspective law, registration polygon, clock overlay.
3. Run steps 3–7. Keep every hard limit in the prompts.

---

## 4. Checks and known limitations

Checked on every kept take, in the final fixed view:
- **The script line:** the take shows what the script's "what the camera sees" text describes.
- **Hard limits:**
  - Adults only and nothing sexual: the attempt is a grab of her arm and a lunge towards her face while her face is
    turned away, and their faces never touch.
  - No kiss, no groping, no hand on her chest or under her coat, no nudity.
  - Her response is a two-handed shove, not a strike.
  - No weapons, no injuries, no other people.
- **Camera stability:** bench and railing within about 1–2 px of the plate (`parallax.py`).
- **Beat boundaries:** no visible jump.
- **Size:** each person's size follows the perspective law.

Every video except the clean version carries a SYNTHETIC tag.

Known limitations:
- 16 fps.
- Faces are generated and can look synthetic in the closest beats.
- In a few bench seconds the man leans back further than a real person would.
- The jacket colour drifts slightly over the 90 s.

---

## 5. Repository layout and script reference

```
scripts/   pipeline code (every server path comes from scripts/sg_paths.py)
slurm/     wan_worker.sbatch, plate.sbatch, submit.sh (account/partition/QOS from config/env.sh)
config/    env.example.sh (copy to env.sh, git-ignored), load_env.sh (sourced by every shell script)
setup/     install_envs.sh, download_weights.sh, check_setup.py, data_bundle.sh, relocate_paths.py
env/       requirements for the three environments; reference pip freezes of the original server
records/   chain.json (kept take, prompt, lock stats per beat), kept_takes.json (exact request per kept take)
docs/      NEW_SERVER.md, PIPELINE.md, POC_REPORT.md
```

| Script | What it does |
|---|---|
| `story.py` | Beat table (id, time, panel, behaviour, tier, script text) and the generation prompt per beat |
| `wan_worker.py` | GPU worker: loads Wan2.2 + merges LightX2V, serves `work/queue` (takes and `cmd` requests) |
| `submit.py` | Queue takes for a beat (seeds, `--prompt`/`--extra`, `--last` for pinned beats) |
| `replay.py` | Re-queue the exact request of a kept take from `records/kept_takes.json` |
| `make_real_plate.py` | Real camera frame → plate (timestamp overlay inverted) |
| `take_frame.py` | One frame of a take as PNG, optionally registered to the plate, optionally with a SAM3 mask |
| `move_seated.py` | Opening frame: move a seated cut-out along the bench with perspective scaling and bench occlusion |
| `chain.py` | Chain bookkeeping (`init`, `status`) |
| `rev.py`, `review.py`, `grid.py` | Review grids of a beat's takes in the fixed view, with camera movement per take |
| `parallax.py`, `motion.py`, `sizecheck.py`, `sizecheck2.py` | Checks: bench/railing alignment, background drift, size stability |
| `commit.sh` | Commit a take: `lockjob.py` → `sam3_instances.py` → `compose.py` → `colorfix.py seed` |
| `lockjob.py`, `bglock.py` | Ground-plane registration to the plate, SAM3 person masks, background lock |
| `compose.py` | Final per-beat composite: mattes, geometric cast shadows, real plate |
| `colorfix.py` | Per-garment colour measurement and correction; writes the next beat's start frame |
| `end_from_take.py`, `pin_end.py`, `make_keyframe.py` | End frames for pinned beats |
| `blend_tail.py` | Ease a regenerated beat into the untouched next beat |
| `sam3_people.py`, `sam3_instances.py`, `sam3_instance.py`, `sam3_garments.py` | SAM3 jobs (run in the SAM3 env on the worker) |
| `compose_all.sh` | Re-compose all beats and write the 720p masters |
| `sr_stage.py` | SeedVR2 1080p upscale (`queue`) and 1080p re-lock (`relock`) |
| `render_final.sh`, `assemble.py`, `beatsheet.py` | Deliverable videos (annotated / CCTV / clean) and beat sheets |
| `run_worker_local.sh` | Start the worker without SLURM |
| `make_start.py`, `make_proxy.py`, `clean_osd.py`, `make_plate.py` | Z-Image-Turbo tools from the first version (stand-in platform, inpainted opening frame, OSD inpainting); not used in the delivered version |
