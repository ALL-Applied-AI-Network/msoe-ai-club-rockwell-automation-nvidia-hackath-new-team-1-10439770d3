#!/usr/bin/env bash
set -u

ROOT=/home/ubuntu/orbit_jobs
VENV="$ROOT/venvs/dvlt"
REPO="$ROOT/repos/dvlt"
RESULT="$ROOT/results/mapanything_offline18_clean_torch"
IMAGES="$ROOT/work/outputs/physical_orbit_reconstruction_20261027/inputs_orbit/orbit_endpoints_18/raw"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$ROOT/huggingface"
export UV_CACHE_DIR="$ROOT/uv-cache"
mkdir -p "$RESULT"
source "$VENV/bin/activate"

START=$(date +%s)
{
  echo "START=$(date --iso-8601=seconds)"
  # Restore the CUDA 12.4 stack on which DVLT was tested.  Installing
  # uniception 0.1.7 with dependencies had pulled an incompatible future
  # CUDA 13 torch build into the first retry.
  uv pip install --reinstall torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124
  uv pip install --reinstall uniception==0.1.7 --no-deps
  uv pip install --reinstall 'git+https://github.com/facebookresearch/map-anything.git' --no-deps
  python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
  cd "$REPO"
  python -m dvlt.scripts.gradio_demo \
    --offline --input "$IMAGES" --models mapanything \
    --output-root "$RESULT/outputs" --max-points 1500000 \
    --conf-threshold 25 --frustum-scale 0 --overwrite
  RC=$?
  echo "RUN_RC=$RC"
} 2>&1 | tee "$RESULT/mapanything.log"
RC=${PIPESTATUS[0]}
END=$(date +%s)
echo $((END - START)) > "$RESULT/runtime_seconds.txt"
echo "$RC" > "$RESULT/returncode.txt"

{
  git -C "$REPO" rev-parse HEAD
  find "$RESULT" -maxdepth 6 -type f -printf '%p %s bytes\n'
  df -h "$ROOT"
} | tee "$RESULT/environment_and_files.txt"

tar -czf "$ROOT/results/mapanything_offline18_clean_torch_artifacts.tar.gz" \
  -C "$ROOT/results" mapanything_offline18_clean_torch
sha256sum "$ROOT/results/mapanything_offline18_clean_torch_artifacts.tar.gz"
exit "$RC"
