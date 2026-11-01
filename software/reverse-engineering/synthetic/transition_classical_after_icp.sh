#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shadeform/work/isaac_run
CONTAINER=re2dgs118

until [ -f "$ROOT/results/gearboxopen/reconstructions/tsdf_icp.json" ] && \
      [ -f "$ROOT/results/gearboxclosed/reconstructions/tsdf_icp.json" ]; do
  sleep 20
done

# Preserve the completed 90-view FK and bounded-ICP results, then replace the
# expensive all-method parent jobs with a controlled, identical 36-view budget
# for the remaining candidates.
docker exec "$CONTAINER" pkill -f 'reconstruct_classical.py --dataset /work/isaac_run/datasets/gearboxopen' || true
docker exec "$CONTAINER" pkill -f 'reconstruct_classical.py --dataset /work/isaac_run/datasets/gearboxclosed' || true
sleep 2

METHODS='poisson_depth laser_stripe_poisson visual_hull nbv_tsdf dual_pass_tsdf photometric_poisson touch_scaled_tsdf'
docker exec -d "$CONTAINER" bash -lc \
  "taskset -c 0-3 python /work/isaac_run/input/reconstruct_classical.py \
    --dataset /work/isaac_run/datasets/gearboxclosed \
    --out /work/isaac_run/results/gearboxclosed/reconstructions \
    --methods $METHODS --max-frames 36 --voxel-mm 0.75 \
    --visual-hull-mm 1.25 --poisson-depth 9 --nbv-views 24 \
    > /work/isaac_run/results/gearboxclosed/classical_remaining.log 2>&1"

docker exec -d "$CONTAINER" bash -lc \
  "taskset -c 4-7 python /work/isaac_run/input/reconstruct_classical.py \
    --dataset /work/isaac_run/datasets/gearboxopen \
    --out /work/isaac_run/results/gearboxopen/reconstructions \
    --methods $METHODS --max-frames 36 --voxel-mm 0.75 \
    --visual-hull-mm 1.25 --poisson-depth 9 --nbv-views 24 \
    > /work/isaac_run/results/gearboxopen/classical_remaining.log 2>&1"

printf 'started %s\n' "$(date -Iseconds)" > "$ROOT/results/classical_remaining.started"
