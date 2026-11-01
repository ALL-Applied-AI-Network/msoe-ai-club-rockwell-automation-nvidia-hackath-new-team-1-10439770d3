#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RESULT_NAME INPUT_FOLDER" >&2
  exit 2
fi

NAME="$1"
INPUT_FOLDER="$2"
ROOT=/home/ubuntu/orbit_jobs
VENV="$ROOT/venvs/blast3r"
REPO="$ROOT/repos/blast3r"
RESULT="$ROOT/results/$NAME"
IMAGES="$ROOT/work/outputs/physical_orbit_reconstruction_20261027/inputs_orbit/$INPUT_FOLDER/raw"
MASKS="$ROOT/work/outputs/physical_orbit_reconstruction_20261027/inputs_orbit/$INPUT_FOLDER/masks"
EXPORTER="$ROOT/work/physical_orbit_reconstruction/export_blast3r_scene.py"

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
export HF_HOME="$ROOT/huggingface"
export UV_CACHE_DIR="$ROOT/uv-cache"
mkdir -p "$RESULT"
source "$VENV/bin/activate"

{
  date --iso-8601=seconds
  git -C "$REPO" rev-parse HEAD
  python --version
  python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
  find "$IMAGES" -maxdepth 1 -type f | wc -l
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
  --confidence-percentile 25 \
  --max-points 1500000

tar -czf "$ROOT/results/${NAME}_artifacts.tar.gz" \
  -C "$ROOT/results" \
  "$NAME/environment.txt" \
  "$NAME/inference.log" \
  "$NAME/runtime_seconds.txt" \
  "$NAME/blast3r_masked_conf25.ply" \
  "$NAME/blast3r_masked_conf25.json"
sha256sum "$ROOT/results/${NAME}_artifacts.tar.gz"
