#!/usr/bin/env bash
set -euo pipefail

# Stage-0 TripoSR fine-tuning environment.
# The official TripoSR repo is inference-first; this script installs it so we
# can inspect module names and then add a real chair fine-tuning loop.

VENV_DIR="${VENV_DIR:-/data/venv}"
TRIPOSR_DIR="${TRIPOSR_DIR:-/data/TripoSR}"

source "${VENV_DIR}/bin/activate"

if [ ! -d "${TRIPOSR_DIR}" ]; then
  echo "[triposr] cloning official repo into ${TRIPOSR_DIR}"
  git clone https://github.com/VAST-AI-Research/TripoSR.git "${TRIPOSR_DIR}"
fi

cd "${TRIPOSR_DIR}"

python -m pip install --upgrade setuptools wheel
python -m pip install --root-user-action=ignore -r requirements.txt
python -m pip install --root-user-action=ignore git+https://github.com/tatsy/torchmcubes.git

echo "[triposr] verifying import"
PYTHONPATH="${TRIPOSR_DIR}:${PYTHONPATH:-}" python - <<'PY'
from tsr.system import TSR
print("TSR import ok:", TSR)
PY

echo "[triposr] done"
