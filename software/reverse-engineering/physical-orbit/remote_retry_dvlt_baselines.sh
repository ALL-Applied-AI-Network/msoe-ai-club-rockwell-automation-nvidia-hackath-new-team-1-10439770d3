#!/usr/bin/env bash
set -u

ROOT=/home/ubuntu/orbit_jobs
VENV="$ROOT/venvs/dvlt"
REPO="$ROOT/repos/dvlt"
RESULT="$ROOT/results/dvlt_baselines_retry_offline18"
IMAGES="$ROOT/work/outputs/physical_orbit_reconstruction_20261027/inputs_orbit/orbit_endpoints_18/raw"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$ROOT/huggingface"
export UV_CACHE_DIR="$ROOT/uv-cache"
mkdir -p "$RESULT"
source "$VENV/bin/activate"

run_candidate() {
  local name="$1"
  local install="$2"
  local log="$RESULT/${name}.log"
  local started ended rc install_rc
  started=$(date +%s)
  {
    echo "CANDIDATE=$name"
    echo "START=$(date --iso-8601=seconds)"
    eval "$install"
    install_rc=$?
    echo "INSTALL_RC=$install_rc"
    if [[ $install_rc -eq 0 ]]; then
      cd "$REPO"
      python -m dvlt.scripts.gradio_demo \
        --offline --input "$IMAGES" --models "$name" \
        --output-root "$RESULT/outputs" --max-points 1500000 \
        --conf-threshold 25 --frustum-scale 0 --overwrite
      rc=$?
    else
      rc=$install_rc
    fi
    echo "RUN_RC=$rc"
  } 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  ended=$(date +%s)
  echo $((ended - started)) > "$RESULT/${name}_runtime_seconds.txt"
  echo "$rc" > "$RESULT/${name}_returncode.txt"
  return 0
}

# The first audit run deliberately used the upstream no-deps command from
# DVLT's install guide.  VGGT also imports these lightweight runtime packages.
run_candidate vggt "uv pip install einops safetensors && uv pip install 'git+https://github.com/facebookresearch/vggt.git' --no-deps"

# MapAnything 1.1.4 declares uniception 0.1.7.  DVLT's pinned 0.1.6 guide is
# stale for the current upstream and rejects the current encoder config.
run_candidate mapanything "uv pip install --upgrade uniception==0.1.7 && uv pip install 'git+https://github.com/facebookresearch/map-anything.git' --no-deps"

{
  git -C "$REPO" rev-parse HEAD
  python --version
  python - <<'PY'
import torch
import einops
import uniception
print('torch', torch.__version__)
print('cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('einops', einops.__version__)
print('uniception', getattr(uniception, '__version__', 'unknown'))
PY
  find "$RESULT" -maxdepth 6 -type f -printf '%p %s bytes\n'
  df -h "$ROOT"
} | tee "$RESULT/environment_and_files.txt"

tar -czf "$ROOT/results/dvlt_baselines_retry_offline18_artifacts.tar.gz" \
  -C "$ROOT/results" dvlt_baselines_retry_offline18
sha256sum "$ROOT/results/dvlt_baselines_retry_offline18_artifacts.tar.gz"
