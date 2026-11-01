#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/orbit_jobs
VENV="$ROOT/venvs/dvlt"
REPO="$ROOT/repos/dvlt"
RESULT="$ROOT/results/dvlt_offline18"
IMAGES="$ROOT/work/outputs/physical_orbit_reconstruction_20261027/inputs_orbit/orbit_endpoints_18/raw"

export PATH="$HOME/.local/bin:$PATH"
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

if ! python -c 'import torch; assert torch.__version__.startswith("2.5.1")' >/dev/null 2>&1; then
  uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
fi
if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 https://github.com/nv-tlabs/dvlt.git "$REPO"
fi
uv pip install -e "$REPO[demos]"

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
python -m dvlt.scripts.gradio_demo \
  --offline \
  --input "$IMAGES" \
  --models dvlt \
  --output-root "$RESULT/outputs" \
  --max-points 1500000 \
  --conf-threshold 25 \
  --frustum-scale 0 \
  --overwrite 2>&1 | tee "$RESULT/inference.log"
END=$(date +%s)
echo $((END - START)) > "$RESULT/runtime_seconds.txt"

find "$RESULT" -maxdepth 4 -type f -printf '%p %s bytes\n' | tee "$RESULT/files.txt"
tar -czf "$ROOT/results/dvlt_offline18_artifacts.tar.gz" \
  -C "$ROOT/results" dvlt_offline18
sha256sum "$ROOT/results/dvlt_offline18_artifacts.tar.gz"
