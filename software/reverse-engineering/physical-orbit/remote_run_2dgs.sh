#!/usr/bin/env bash
set -u

ROOT=/home/ubuntu/orbit_jobs
VENV="$ROOT/venvs/dvlt"
REPO="$ROOT/repos/2d-gaussian-splatting"
DATA="$ROOT/work/nerfstudio_orbit18"
RESULT="$ROOT/results/2dgs_colmap18"
export PATH="$HOME/.local/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=8.9
export UV_CACHE_DIR="$ROOT/uv-cache"
mkdir -p "$RESULT" "$ROOT/repos"
source "$VENV/bin/activate"

START=$(date +%s)
{
  echo "START=$(date --iso-8601=seconds)"
  if [[ ! -d "$REPO/.git" ]]; then
    git clone --recursive https://github.com/hbb1/2d-gaussian-splatting.git "$REPO"
  else
    git -C "$REPO" submodule update --init --recursive
  fi
  # CUDA 12.8 no longer exposes FLT_MAX transitively in this older extension.
  # Keep the official implementation unchanged except for the standard header
  # it directly requires, then discard the failed partial build before retry.
  KNN_CU="$REPO/submodules/simple-knn/simple_knn.cu"
  if ! grep -q '^#include <cfloat>' "$KNN_CU"; then
    sed -i '1i#include <cfloat>' "$KNN_CU"
  fi
  rm -rf "$REPO/submodules/simple-knn/build"
  uv pip install plyfile tqdm opencv-python
  uv pip install --no-build-isolation "$REPO/submodules/diff-surfel-rasterization"
  uv pip install --no-build-isolation "$REPO/submodules/simple-knn"
  cd "$REPO"
  python train.py -s "$DATA" -m "$RESULT/model" --iterations 7000 --quiet
  TRAIN_RC=$?
  echo "TRAIN_RC=$TRAIN_RC"
  if [[ $TRAIN_RC -eq 0 ]]; then
    python render.py -s "$DATA" -m "$RESULT/model" \
      --iteration 7000 --skip_train --skip_test --mesh_res 512
    RENDER_RC=$?
  else
    RENDER_RC=3
  fi
  echo "RENDER_RC=$RENDER_RC"
} 2>&1 | tee "$RESULT/run.log"
RC=${PIPESTATUS[0]}
END=$(date +%s)
echo $((END - START)) > "$RESULT/runtime_seconds.txt"
echo "$RC" > "$RESULT/returncode.txt"

{
  git -C "$REPO" rev-parse HEAD 2>/dev/null || true
  python --version
  python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
  find "$RESULT" -maxdepth 8 -type f -printf '%p %s bytes\n'
  df -h "$ROOT"
} | tee "$RESULT/environment_and_files.txt"

tar -czf "$ROOT/results/2dgs_colmap18_artifacts.tar.gz" \
  --exclude='*.pth' --exclude='*.pt' \
  -C "$ROOT/results" 2dgs_colmap18
sha256sum "$ROOT/results/2dgs_colmap18_artifacts.tar.gz"
exit "$RC"
