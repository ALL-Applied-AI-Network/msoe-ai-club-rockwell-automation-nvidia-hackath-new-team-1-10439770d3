#!/usr/bin/env bash
set -euo pipefail

# Run this on the team's provisioned DGX Spark workspace after SSH login.
# It pulls the exact Isaac Sim version used to validate Dataset 3 and keeps all
# persistent files beneath /workspace.

mkdir -p /workspace/isaac-cache/{kit,ov,pip,glcache,computecache} /workspace/ateam

docker pull nvcr.io/nvidia/isaac-sim:6.0.0

docker run --rm --gpus all -u 0 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -v /workspace:/workspace \
  -v /workspace/isaac-cache/kit:/isaac-sim/kit/cache:rw \
  -v /workspace/isaac-cache/ov:/root/.cache/ov:rw \
  -v /workspace/isaac-cache/pip:/root/.cache/pip:rw \
  -v /workspace/isaac-cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v /workspace/isaac-cache/computecache:/root/.nv/ComputeCache:rw \
  nvcr.io/nvidia/isaac-sim:6.0.0 \
  /isaac-sim/python.sh -c "from isaacsim import SimulationApp; app=SimulationApp({'headless': True}); print('ISAAC_SIM_6_OK'); app.close()"

echo "Isaac Sim 6.0 image pulled and headless smoke test passed."

STARTER=/workspace/fanuc_er4ia_sim
if [ -f "$STARTER/build_scene.py" ]; then
  mkdir -p "$STARTER/runs"
  docker run --rm --gpus all -u 0 \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -v /workspace:/workspace \
    -v /workspace/isaac-cache/kit:/isaac-sim/kit/cache:rw \
    -v /workspace/isaac-cache/ov:/root/.cache/ov:rw \
    -v /workspace/isaac-cache/pip:/root/.cache/pip:rw \
    -v /workspace/isaac-cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v /workspace/isaac-cache/computecache:/root/.nv/ComputeCache:rw \
    -w "$STARTER" \
    nvcr.io/nvidia/isaac-sim:6.0.0 \
    /isaac-sim/python.sh build_scene.py --headless --save runs/point_to_point_scene.usd
  test -s "$STARTER/runs/point_to_point_scene.usd"
  echo "Dataset 3 scene built at $STARTER/runs/point_to_point_scene.usd"
else
  echo "Dataset 3 is not in $STARTER yet; copy it there and rerun this script."
fi
