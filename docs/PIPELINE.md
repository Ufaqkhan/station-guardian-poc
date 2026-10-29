# Pipeline: commands step by step

All commands run from the data folder with the environment loaded:

```bash
source config/load_env.sh     # from the repository; sets SG_REPO, SG_ROOT, SG_PY, SG_SAM3_PY, ...
cd "$SG_ROOT"
S=$SG_REPO/scripts
```

Paths given on the command line (`work/...`, `chain/...`) are relative to `$SG_ROOT`.

## 0. The GPU worker

Everything that needs a GPU goes through one worker: Wan2.2 takes, plus SAM3 and SeedVR2 as "cmd" requests (SAM3
hard-codes CUDA, so it cannot run on the CPU side).

```bash
slurm/submit.sh wan                        # SLURM, 4 h default; add e.g. --time=12:00:00
GPU=0 scripts/run_worker_local.sh          # no SLURM
touch "$SG_ROOT/work/queue/STOP"           # stop all workers after their current request
```

The queue is the folder `work/queue/`: one JSON per request, claimed by atomic rename into `running/`, then moved to
`done/` or `failed/`. Several workers can share one queue. Lower `priority` runs first.

## 1. Plate and opening frame (already done for GP02; needed only for a new camera view)

| What | Command |
|---|---|
| Plate: remove the burned-in timestamp from the real frame | `$SG_PY $S/make_real_plate.py --src View-Display.jpeg` |
| Opening frame: walk-in take, frame + SAM3 mask, place on the seat | `submit.py --beat P0a --tag g1 --start work/plates/plate.png --seeds 1 2 3`, `take_frame.py ... --idx 80 --mask "woman in a red coat sitting on a bench"`, `move_seated.py --frac 0.20` (full commands in the README, section 2.3) |
| (first version, optional) inpaint with Z-Image-Turbo | `make_proxy.py` (scaled stand-in) then `make_start.py --init ... --strength 0.82`; `clean_osd.py` |
| Register plate and start in the chain | `$SG_PY $S/chain.py init --plate work/plates/plate.png --start work/start/real/A01_start.png` |

## 2. One beat

```bash
# a. takes: one request per seed, from the previous beat's corrected last frame
$SG_PY $S/submit.py --beat A06 --start chain/A05_last_cc.png --seeds 1 2 3 4 5 6 --tag g1
#    prompt = story.py prompt for the beat; --extra "..." appends, --prompt "..." replaces

#    or re-run the exact request of the delivered take:  $SG_PY $S/replay.py A06 [--seeds ...]

# b. review in the final fixed view (registered to the plate), pick the take
$SG_PY $S/rev.py A06 g1 --n 6 --box 790 110 1070 400 --idx 16 32 48 64 80     # grid + parallax per take
$SG_PY $S/parallax.py work/beats/A06/g1/A06_g1_s3.mkv                         # bench/railing edge error

# c. commit: registration + SAM3 masks + composite with shadows + colour correction + seed for the next beat
scripts/commit.sh A06 work/beats/A06/g1/A06_g1_s3.mkv
```

Outputs per beat: `chain/A06.mkv` (composited, lossless), `chain/A06_last.png`, `chain/A06_last_cc.png` (colour
corrected, start of the next beat), `work/lock/A06/` (homographies, masks, matte), `work/colorfix/A06_*`.

### Pinned beats (when the model pans, zooms, or drifts into a wrong pose)

Build the end frame in the fixed view, then generate the motion between the two frames (first + last frame mode):

```bash
$SG_PY $S/take_frame.py work/beats/A14/g1/A14_g1_s6.mkv --idx 28 --reg --out work/keyframes/real/A14/s6_f28_reg.png
$SG_PY $S/end_from_take.py --src work/keyframes/real/A14/s6_f28_reg.png --out work/keyframes/real/A14/A14_end.png
$SG_PY $S/submit.py --beat A14 --start chain/A13_last_cc.png --last work/keyframes/real/A14/A14_end.png --seeds 1 2 3 4 --tag p1
```

Other end-frame tools: `pin_end.py` (replace one person in the previous last frame), `make_keyframe.py` (cut-outs
moved/scaled by the perspective law). After regenerating a beat in the middle of the chain, `blend_tail.py` eases its
last frames into the untouched next beat.

## 3. Finish

```bash
scripts/compose_all.sh                                   # re-compose all beats, write 720p masters (work/master/)
$SG_PY $S/sr_stage.py queue --master work/master/story_720.mkv --tag story   # SeedVR2 on the worker GPU
$SG_PY $S/sr_stage.py queue --master work/master/twin_720.mkv --tag twin
scripts/render_final.sh                                  # 1080p re-lock, videos, beat sheets -> output/
```

Quick looks while working: `assemble.py --partial` (committed beats so far), `assemble.py --preview 42.5`
(one annotated still).

## Constants measured on the GP02 view

If the camera view changes, these have to be measured again:

- `compose.py`: `SUN_A, SUN_B` (shadow direction), `RATIO` (shadow length), `HORIZON` (person height
  h = 0.75·(foot_y + 36) in 720p pixels), `EDGE` table, shade-map thresholds.
- `bglock.py`: `GROUND` polygon (platform surface used for registration), `BENCH_REGION`.
- `assemble.py`: on-screen clock position and style (`osd()`), the real camera's overlay `obs = 0.32·scene + 10`.
- `move_seated.py`, `make_proxy.py`: bench geometry; `work/plates/real/bench_front_mask.png` (bench parts in front of
  a seated person).

## Lessons (what went wrong and the fix)

- diffusers' Wan LoRA converter drops the LightX2V `diff` / `diff_m` tensors: the adapters are merged by hand in
  `wan_worker.merged_transformer`; `k_img` / `v_img` keys are skipped (absent in Wan2.2 I2V).
- CFG > 1 with the 4-step adapters burns colours: keep guidance 1.0 and steer with prompt wording
  ("remains seated for the whole clip").
- Wan pans or zooms 50–250 px when people stand or walk apart: pick takes by ground homography (`rev.py` prints it);
  when every take moves, pin the end frame.
- Register on the platform surface only (`bglock.ground_homographies`); whole-frame fits ghost the bench.
- The man drifts into reclined or lying poses whenever a prompt says lean or slide; shove beats knock him off the bench
  unless the prompt says "rocks back against the armrest and stays sitting; does not fall".
- Chained image-to-video drifts colours (green → teal, grey hair → blue): correct per garment and seed the next beat
  from the corrected frame; `colorfix.py seed` re-measures when the clip is newer than its offsets.
- Z-Image inpainting into a large box draws people about twice life size: draw a correctly scaled stand-in first and
  refine at strength 0.82.
- Remove a translucent OSD by inverting its measured overlay, not by inpainting.
