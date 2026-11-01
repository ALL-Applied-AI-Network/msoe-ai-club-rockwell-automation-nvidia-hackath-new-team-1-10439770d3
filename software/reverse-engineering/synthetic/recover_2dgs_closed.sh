#!/usr/bin/env bash
set -u

ROOT=/work/isaac_run
REPO=/work/2d-gaussian-splatting
SOURCE="$ROOT/official_inputs/colmap/gearboxclosed"
MODEL="$ROOT/official_results/2dgs/gearboxclosed"
mkdir -p "$MODEL"
rm -f "$ROOT/official_results/2dgs/all.done"
cd "$REPO"

# The first training reached 7000/7000 but Matplotlib 3.10 removed
# FigureCanvasAgg.tostring_rgb during the optional evaluation render before the
# checkpoint was written.  Skip that reporting-only callback and preserve the
# actual training/save path.
start=$(date +%s)
python train.py \
  -s "$SOURCE" -m "$MODEL" --images images --resolution 1 \
  --iterations 7000 --save_iterations 7000 --test_iterations 999999 \
  --lambda_dist 1000 --lambda_normal 0.05 \
  > "$MODEL/train_recovery.log" 2>&1
status=$?
printf '%s\n' "$status" > "$MODEL/train_recovery.exit"
printf '%s\n' "$(( $(date +%s) - start ))" > "$MODEL/train_recovery_wall_seconds.txt"

if [ "$status" -eq 0 ]; then
  start=$(date +%s)
  python render.py \
    -s "$SOURCE" -m "$MODEL" --images images --iteration 7000 \
    --skip_test --voxel_size 0.00125 --depth_trunc 0.60 \
    --sdf_trunc 0.004 --num_cluster 20 \
    > "$MODEL/render_bounded.log" 2>&1
  printf '%s\n' "$?" > "$MODEL/render_bounded.exit"
  printf '%s\n' "$(( $(date +%s) - start ))" > "$MODEL/render_wall_seconds.txt"
fi
printf 'complete\n' > "$ROOT/official_results/2dgs/all.done"
