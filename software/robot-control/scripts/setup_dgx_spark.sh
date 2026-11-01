#!/usr/bin/env bash
set -euo pipefail

# Run this from the team's provisioned DGX Spark workspace after SSH login.
# YAAD pool jobs mount that workspace at /workspace, while an interactive SSH
# session exposes it as the directory containing this script. Mount the actual
# persistent directory into the Isaac container at /workspace in both cases.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKSPACE_ROOT="${ATEAM_WORKSPACE_ROOT:-$SCRIPT_DIR}"
mkdir -p "$WORKSPACE_ROOT"/isaac-cache/{kit,ov,pip,glcache,computecache} "$WORKSPACE_ROOT/ateam"

STARTER="$WORKSPACE_ROOT/fanuc_er4ia_sim"
if [ ! -f "$STARTER/build_scene.py" ]; then
  echo "Dataset 3 is not in $STARTER." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    python3 python3-venv python3-pip
fi

cd "$STARTER"
chmod +x bootstrap.sh
./bootstrap.sh

docker pull nvcr.io/nvidia/isaac-sim:6.0.0

docker run --rm --gpus all -u 0 \
  --entrypoint /isaac-sim/python.sh \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e OMNI_KIT_ALLOW_ROOT=1 \
  -v "$WORKSPACE_ROOT:/workspace" \
  -v "$WORKSPACE_ROOT/isaac-cache/kit:/isaac-sim/kit/cache:rw" \
  -v "$WORKSPACE_ROOT/isaac-cache/ov:/root/.cache/ov:rw" \
  -v "$WORKSPACE_ROOT/isaac-cache/pip:/root/.cache/pip:rw" \
  -v "$WORKSPACE_ROOT/isaac-cache/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "$WORKSPACE_ROOT/isaac-cache/computecache:/root/.nv/ComputeCache:rw" \
  nvcr.io/nvidia/isaac-sim:6.0.0 \
  -c "from isaacsim import SimulationApp; app=SimulationApp({'headless': True}); print('ISAAC_SIM_6_OK'); app.close()"

echo "Isaac Sim 6.0 image pulled and headless smoke test passed."

mkdir -p "$STARTER/runs"
docker run --rm --gpus all -u 0 \
  --entrypoint /isaac-sim/python.sh \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e OMNI_KIT_ALLOW_ROOT=1 \
  -v "$WORKSPACE_ROOT:/workspace" \
  -v "$WORKSPACE_ROOT/isaac-cache/kit:/isaac-sim/kit/cache:rw" \
  -v "$WORKSPACE_ROOT/isaac-cache/ov:/root/.cache/ov:rw" \
  -v "$WORKSPACE_ROOT/isaac-cache/pip:/root/.cache/pip:rw" \
  -v "$WORKSPACE_ROOT/isaac-cache/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "$WORKSPACE_ROOT/isaac-cache/computecache:/root/.nv/ComputeCache:rw" \
  -w /workspace/fanuc_er4ia_sim \
  nvcr.io/nvidia/isaac-sim:6.0.0 \
  build_scene.py --headless --save runs/point_to_point_scene.usd
test -s "$STARTER/runs/point_to_point_scene.usd"
echo "Dataset 3 scene built at $STARTER/runs/point_to_point_scene.usd"

docker run --rm --gpus all -u 0 \
  --entrypoint /isaac-sim/python.sh \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e OMNI_KIT_ALLOW_ROOT=1 \
  -v "$WORKSPACE_ROOT:/workspace" \
  -v "$WORKSPACE_ROOT/isaac-cache/kit:/isaac-sim/kit/cache:rw" \
  -v "$WORKSPACE_ROOT/isaac-cache/ov:/root/.cache/ov:rw" \
  -v "$WORKSPACE_ROOT/isaac-cache/pip:/root/.cache/pip:rw" \
  -v "$WORKSPACE_ROOT/isaac-cache/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "$WORKSPACE_ROOT/isaac-cache/computecache:/root/.nv/ComputeCache:rw" \
  -w /workspace/fanuc_er4ia_sim \
  nvcr.io/nvidia/isaac-sim:6.0.0 \
  isaac_point_to_point.py --out runs/point_to_point_report.json

"$STARTER/.venv/bin/python" - "$STARTER/runs/point_to_point_report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report.get("task_complete") is True, report
assert len(report.get("waypoints", [])) == 2, report
print("Point-to-point report verified:", sys.argv[1])
PY
