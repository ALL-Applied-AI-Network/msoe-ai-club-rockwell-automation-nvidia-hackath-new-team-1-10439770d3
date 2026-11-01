#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/orbit_jobs
VENV="$ROOT/venvs/blast3r"
REPO="$ROOT/repos/blast3r"
RESULT="$ROOT/results/blast3r_offline18"
IMAGES="$ROOT/work/outputs/physical_orbit_reconstruction_20261027/inputs_orbit/orbit_endpoints_18/raw"
MASKS="$ROOT/work/outputs/physical_orbit_reconstruction_20261027/inputs_orbit/orbit_endpoints_18/masks"
EXPORTER="$ROOT/work/physical_orbit_reconstruction/export_blast3r_scene.py"

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
export HF_HOME="$ROOT/huggingface"
export UV_CACHE_DIR="$ROOT/uv-cache"
mkdir -p "$ROOT/venvs" "$ROOT/repos" "$RESULT" "$HF_HOME"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv python install 3.11
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python 3.11 "$VENV"
fi
source "$VENV/bin/activate"

if ! python -c 'import torch' >/dev/null 2>&1; then
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi
uv pip install ninja packaging wheel setuptools
if ! python -c 'import torch_scatter' >/dev/null 2>&1; then
  uv pip install --no-build-isolation torch-scatter
fi
if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 https://github.com/naver/blast3r.git "$REPO"
fi
uv pip install --no-build-isolation -e "$REPO"
if ! python -c 'import faiss' >/dev/null 2>&1; then
  uv pip install faiss-gpu-cu12 || uv pip install faiss-cpu
fi

{
  date --iso-8601=seconds
  git -C "$REPO" rev-parse HEAD
  python --version
  python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} | tee "$RESULT/environment.txt"

START=$(date +%s)
cd "$REPO"
python scripts/run_offline.py \
  --images "$IMAGES" \
  --output "$RESULT/scenes" \
  --mono_cam \
  --denoise_depth 2>&1 | tee "$RESULT/inference.log"
END=$(date +%s)
echo $((END - START)) > "$RESULT/runtime_seconds.txt"

SCENE=$(find "$RESULT/scenes" -type f -name '*.pth' | head -1)
if [[ -z "$SCENE" ]]; then
  echo 'BLASt3R produced no .pth scene' >&2
  exit 3
fi
python "$EXPORTER" \
  --scene "$SCENE" \
  --masks "$MASKS" \
  --output "$RESULT/blast3r_masked_conf25.ply" \
  --confidence-percentile 25

tar -czf "$ROOT/results/blast3r_offline18_artifacts.tar.gz" \
  -C "$ROOT/results" \
  blast3r_offline18/environment.txt \
  blast3r_offline18/inference.log \
  blast3r_offline18/runtime_seconds.txt \
  blast3r_offline18/blast3r_masked_conf25.ply \
  blast3r_offline18/blast3r_masked_conf25.json
sha256sum "$ROOT/results/blast3r_offline18_artifacts.tar.gz"
