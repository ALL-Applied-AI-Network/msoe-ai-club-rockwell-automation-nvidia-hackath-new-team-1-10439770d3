#!/usr/bin/env bash
set -u

ROOT=/home/ubuntu/orbit_jobs
VENV="$ROOT/venvs/dvlt"
DATA="$ROOT/work/nerfstudio_orbit18"
RESULT="$ROOT/results/nerfacto_colmap18"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$ROOT/huggingface"
export UV_CACHE_DIR="$ROOT/uv-cache"
mkdir -p "$RESULT"
source "$VENV/bin/activate"

# Nerfstudio's COLMAP parser defaults to colmap/sparse/0, whereas the original
# COLMAP/3DGS layout uses sparse/0.  Preserve the source data and expose the
# expected alias.
mkdir -p "$DATA/colmap"
if [[ ! -e "$DATA/colmap/sparse" ]]; then
  ln -s ../sparse "$DATA/colmap/sparse"
fi

START=$(date +%s)
{
  echo "START=$(date --iso-8601=seconds)"
  uv pip install 'git+https://github.com/nerfstudio-project/nerfstudio.git'
  # Nerfstudio's current pil_to_numpy implementation is incompatible with the
  # Pillow 12 core ABI on this image.  Pillow 10.4 is the newest stable series
  # with the expected ImagingCore.setimage signature.
  uv pip install 'pillow==10.4.0'
  rm -rf "$RESULT/training" "$RESULT/export_pointcloud" "$RESULT/export_poisson"
  cd "$ROOT"
  ns-train nerfacto \
    --data "$DATA" \
    --output-dir "$RESULT/training" \
    --max-num-iterations 3000 \
    --pipeline.model.predict-normals True \
    --vis tensorboard \
    colmap
  TRAIN_RC=$?
  echo "TRAIN_RC=$TRAIN_RC"
  CONFIG=$(find "$RESULT/training" -type f -name config.yml | sort | tail -1)
  echo "CONFIG=$CONFIG"
  if [[ -n "$CONFIG" ]]; then
    ns-export pointcloud \
      --load-config "$CONFIG" \
      --output-dir "$RESULT/export_pointcloud" \
      --num-points 1000000 \
      --remove-outliers True
    POINT_RC=$?
    echo "POINT_RC=$POINT_RC"
    ns-export poisson \
      --load-config "$CONFIG" \
      --output-dir "$RESULT/export_poisson" \
      --num-points 1000000 \
      --target-num-faces 200000
    POISSON_RC=$?
    echo "POISSON_RC=$POISSON_RC"
  else
    POINT_RC=3
    POISSON_RC=3
  fi
} 2>&1 | tee "$RESULT/run.log"
RC=${PIPESTATUS[0]}
END=$(date +%s)
echo $((END - START)) > "$RESULT/runtime_seconds.txt"
echo "$RC" > "$RESULT/returncode.txt"

{
  python --version
  python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
  find "$RESULT" -maxdepth 8 -type f -printf '%p %s bytes\n'
  df -h "$ROOT"
} | tee "$RESULT/environment_and_files.txt"

# Exclude checkpoints to keep the evidence package compact; exported geometry,
# config, logs, transforms, and metrics remain reproducible from the run.
tar -czf "$ROOT/results/nerfacto_colmap18_artifacts.tar.gz" \
  --exclude='*.ckpt' --exclude='*.pt' \
  -C "$ROOT/results" nerfacto_colmap18
sha256sum "$ROOT/results/nerfacto_colmap18_artifacts.tar.gz"
exit "$RC"
