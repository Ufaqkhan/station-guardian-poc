# Station Guardian PoC: Story A (BTP Scenario 1), Gidea Park Platform 4, GP02

> Report of the delivered PoC (v2, 2026-10-22), copied from the original working folder. Files named here
> (`output/`, `work/`, `chain/`, `View-Display.jpeg`, the script `.docx`) are not in git: see NEW_SERVER.md, step 5.
> Script paths are relative to this repository (`scripts/`, `slurm/`).

A 90-second synthetic CCTV clip that plays the 18 beats (A01–A18) of
`2026-10-16_S1_StoryA_script_for_partner_review.docx` ("The story, beat by beat") on the **real GP02 camera view**
(`View-Display.jpeg`): one fixed camera, two synthetic people, no audio, five seconds per beat. The 35-second benign
twin (A01–A03, then AT1–AT4) is included.

## Deliverables (`output/`)

| File | What it is |
|---|---|
| `StoryA_GP02_annotated.mp4` | **Main PoC video.** 1920x1080, 90 s. CCTV view, beat card (beat, time, storyboard panel, the script's "what the camera sees" text, behaviour label, tier) and a tier timeline |
| `StoryA_GP02_cctv.mp4` | 1920x1080, 90 s. Camera view with the GP02 camera's own on-screen clock (recreated) and a SYNTHETIC tag |
| `StoryA_GP02_clean.mp4` | 1920x1080, 90 s. Camera picture only, no overlays (training-data style) |
| `StoryA_GP02_beat_sheet.jpg` | One page: the middle frame of each beat with id, time, behaviour and tier |
| `StoryA_twin_GP02_{annotated,cctv,clean}.mp4`, `StoryA_twin_GP02_beat_sheet.jpg` | Same for the benign twin (35 s) |

All videos are 16 fps (each beat = 80 frames = 5.0 s). Every frame carries a "SYNTHETIC" tag except the clean
version, so the footage cannot be mistaken for a real incident if it travels.

The previous version, built on a generated stand-in platform before the real view was available, is kept in
`archive/v1_generated_platform/`.

## How it was made

1. **Platform (plate) = the real GP02 frame.** The camera's burned-in timestamp (a translucent dark box, top right)
   was removed by inverting its measured overlay (observed = 0.32 x scene + 10) and inpainting the text strokes, so
   the real scene shows through. `work/plates/plate.png` (1280x720) and `plate_1536.png` (the 1600x900 original size).
2. **Opening frame.** The young woman was generated walking in and sitting down on the real bench (Wan2.2), cut out
   with SAM3 and placed on the middle seat at the size given by the measured perspective; the bench's front rail and
   near armrest stay in front of her. `work/start/real/A01_start.png`.
3. **Beats.** Each beat is one 81-frame image-to-video generation with **Wan2.2-I2V-A14B** (two 14B experts, fp16,
   LightX2V 4-step distillation merged in; ~92 s per take on an H200), starting from the previous beat's last frame,
   with a prompt from that beat's script line plus fixed descriptions of the real scene, the camera and both people
   (`scripts/story.py`). Several takes per beat; each kept take was checked in the final fixed view for the script
   line, the hard limits and camera stability. `chain/chain.json` records take, seed and prompt per beat.
4. **Pinned beats (A02, A03, A04, A10, A13, A14, AT1).** Where the model zoomed after the people or drifted into a
   wrong pose, the end frame was composed in the fixed view (plate + people cut out of a good take frame,
   `scripts/end_from_take.py`, `pin_end.py`) and Wan2.2 generated the motion between the two frames.
5. **Fixed camera.** Every take is registered to the plate on the platform surface (SIFT + RANSAC homography on the
   asphalt and paving); bench and railing then line up within about 1 px (`scripts/parallax.py`).
6. **Composite and shadows (`scripts/compose.py`).** People are cut out with SAM3 (tight 1 px mattes) and placed over
   the real plate. Each person casts a geometric shadow: their silhouette projected onto the platform along the sun
   direction measured on this view (up and to the left, as the lamp-post and generated walk-in shadows show), with
   body depth, a penumbra that widens away from the feet, a contact shadow at the feet, and the plate's own measured
   shade colour. Shadows are not added where the plate is already in shade, so nothing is darkened twice.
7. **Colour stability (`scripts/colorfix.py`).** Each garment (jacket, coat, jeans, chinos, beanie, hair, trainers,
   bag) is segmented with SAM3 and pulled back towards its colour at the start; the corrected last frame seeds the
   next beat.
8. **1080p (`scripts/sr_stage.py`).** The 720p master and the plate are upscaled with SeedVR2-7B; the upscaled people
   are put back over ONE upscaled plate with the same mattes and shadows, so the background has no shimmer. Every
   beat boundary eases over 8 frames into the next beat's first frame.
9. **Assembly.** `scripts/assemble.py` (videos, recreated GP02 clock overlay), `scripts/beatsheet.py`.

Why Wan2.2-I2V-A14B: it is the strongest open image-to-video model for human motion and for following action
descriptions, and the 4-step distillation made several takes per beat affordable.

## Differences from the script (for the partner review)

- **Lighting:** afternoon sun, as in the GP02 frame supplied; the evening version needs evening screenshots.
- **Timestamp:** the clock in the CCTV version is recreated in the camera's own style and runs from 13:36:34 (the
  time on the supplied frame); the date is the frame's date.
- **A15:** after her shove she holds both palms up towards him (warding him off) while he rocks back on the bench.
- **A17:** she backs away up the platform a little further than "two or three steps" (about 4–5 m); A18 ends with the
  two of them well apart, as scripted.
- The man is cast in his fifties as in the first script; the 30s–40s variant is a prompt change in `scripts/story.py`.

## Hard limits (checked on every kept take)

Adults only; nothing sexual: the attempt is a grab of her arm and a lunge towards her face while her face is turned
away, and their faces never touch. No kiss, no groping, no hand on her chest or under her coat, no nudity. Her
response is a two-handed shove, not a strike. No weapons, no injuries, no other people.

## Known limitations

- 16 fps; faces are generated and can look slightly synthetic in the closest beats.
- The generator tends to recline the man in the bench beats; the worst cases were replaced by pinned beats, but a few
  seconds still show him leaning back further than a real person would.
- Small residual colour drift in the jacket over the 90 s despite the correction.

## Revision history

- **v1 (generated platform):** stand-in platform from Z-Image-Turbo; revisions fixed her lower legs, A17/A18 timing,
  her size in the ending, added geometric shadows and 1080p. Archived in `archive/v1_generated_platform/`.
- **v2 (this version):** rebuilt on the real GP02 view supplied by the user, with all shadow and perspective
  constants re-measured on that view (`compose.py`: SUN_A/SUN_B, RATIO, HORIZON, EDGE; `bglock.py`: GROUND).

## Layout

```
scripts/  story.py (beat table + prompts), wan_worker.py + slurm/wan_worker.sbatch (GPU worker; queue in work/queue),
          submit.py, grid.py / rev.py / parallax.py (review), commit.sh (lockjob + SAM3 instances + compose + colour),
          end_from_take.py / pin_end.py / move_seated.py (pinned frames), bglock.py, compose.py, colorfix.py,
          sam3_*.py, sr_stage.py, assemble.py, beatsheet.py, clean_osd.py
work/     plates (real view), opening frame, every take per beat (work/beats/<beat>/<round>/), lock/colour data,
          masters (720p / 1080p)
chain/    the kept take per beat (registered, composited, lossless) and its last frame
output/   final videos and beat sheets
```
