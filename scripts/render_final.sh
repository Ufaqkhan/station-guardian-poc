#!/bin/bash
# Final render (the v2 deliverables): re-lock the SeedVR2 1080p masters, assemble the three videos for the story and
# the twin, write both beat sheets. Run after:  scripts/sr_stage.py queue ... (story and twin) has finished.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config/load_env.sh"
cd "$SG_ROOT"
export CUDA_VISIBLE_DEVICES=""
S="$SG_REPO/scripts"
"$SG_PY" "$S/sr_stage.py" relock --master work/master/story_720.mkv --tag story 2>&1 | grep -E "wrote|Traceback|Error"
"$SG_PY" "$S/sr_stage.py" relock --master work/master/twin_720.mkv --tag twin 2>&1 | grep -E "wrote|Traceback|Error"
"$SG_PY" "$S/assemble.py" --master work/master/story_1080.mkv --out output \
  --stills 2.5 12.5 22.5 32.5 42.5 52.5 62.5 67.5 72.5 77.5 82.5 87.5 2>&1 | grep -v -i warn | tail -2
"$SG_PY" "$S/assemble.py" --twin --master work/master/twin_1080.mkv --prefix StoryA_twin_GP02 --out output 2>&1 \
  | grep -v -i warn | tail -2
"$SG_PY" "$S/beatsheet.py" 2>&1 | tail -1
"$SG_PY" "$S/beatsheet.py" --twin 2>&1 | tail -1
mkdir -p work/qa/final && mv output/still_*.jpg work/qa/final/ 2>/dev/null || true
echo RENDER_DONE
