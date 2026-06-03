#!/usr/bin/env bash
set -euo pipefail

# Reproducible CUDA training environment for the 2xA40 server.
# Usage:
#   bash /data/repositoryi/setup_a40_env.sh
#   source /data/venv/bin/activate

VENV_DIR="${VENV_DIR:-/data/venv}"

echo "[setup] creating venv: ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip uv

echo "[setup] installing PyTorch CUDA 12.4 wheels"
uv pip install --no-cache \
  --index-strategy unsafe-best-match \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple

echo "[setup] installing training dependencies"
uv pip install --no-cache \
  numpy scipy Pillow tqdm pandas trimesh scikit-image matplotlib \
  imageio opencv-python-headless einops omegaconf pyyaml \
  huggingface_hub safetensors

echo "[setup] verifying CUDA"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpus:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

echo "[setup] done"
