# Sourced by every shell script in this repository. Sets SG_REPO, reads config/env.sh, fills defaults.
SG_REPO="${SG_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SG_REPO
if [ -f "$SG_REPO/config/env.sh" ]; then
  source "$SG_REPO/config/env.sh"
else
  echo "note: config/env.sh not found, using config/env.example.sh defaults (cp config/env.example.sh config/env.sh)" >&2
  source "$SG_REPO/config/env.example.sh"
fi
export SG_ROOT="${SG_ROOT:-$SG_REPO}"
export SG_PY="${SG_PY:-python}"
export SG_SAM3_PY="${SG_SAM3_PY:-python}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
