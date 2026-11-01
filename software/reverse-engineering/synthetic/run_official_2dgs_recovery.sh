#!/usr/bin/env bash
set -u

ROOT=/work/isaac_run
REPO=/work/2d-gaussian-splatting

render_part() {
    name="$1"
    source="$ROOT/official_inputs/colmap/$name"
    model="$ROOT/official_results/2dgs/$name"
    cd "$REPO"
    start=$(date +%s)
    python render.py \
        -s "$source" -m "$model" --images images --iteration 7000 \
        --skip_test --voxel_size 0.00125 --depth_trunc 0.60 \
        --sdf_trunc 0.004 --num_cluster 20 \
        > "$model/render_bounded.log" 2>&1
    printf '%s\n' "$?" > "$model/render_bounded.exit"
    printf '%s\n' "$(( $(date +%s) - start ))" > "$model/render_wall_seconds.txt"
}

train_closed() {
    source="$ROOT/official_inputs/colmap/gearboxclosed"
    model="$ROOT/official_results/2dgs/gearboxclosed"
    mkdir -p "$model"
    cd "$REPO"
    start=$(date +%s)
    python train.py \
        -s "$source" -m "$model" --images images --resolution 1 \
        --iterations 7000 --save_iterations 7000 --test_iterations 7000 \
        --lambda_dist 1000 --lambda_normal 0.05 \
        > "$model/train.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$model/train.exit"
    printf '%s\n' "$(( $(date +%s) - start ))" > "$model/train_wall_seconds.txt"
}

render_part gearboxopen
train_closed
if [ "$(cat "$ROOT/official_results/2dgs/gearboxclosed/train.exit")" = 0 ]; then
    render_part gearboxclosed
fi
printf 'complete\n' > "$ROOT/official_results/2dgs/all.done"
