#!/usr/bin/env bash
set -u

ROOT=/work/isaac_run
REPO=/work/2d-gaussian-splatting

run_part() {
    name="$1"
    source="$ROOT/official_inputs/colmap/$name"
    model="$ROOT/official_results/2dgs/$name"
    mkdir -p "$model"
    start=$(date +%s)
    cd "$REPO"
    python train.py \
        -s "$source" -m "$model" --images images --resolution 1 \
        --iterations 7000 --save_iterations 7000 --test_iterations 7000 \
        --lambda_dist 1000 --lambda_normal 0.05 \
        > "$model/train.log" 2>&1
    train_status=$?
    printf '%s\n' "$train_status" > "$model/train.exit"
    if [ "$train_status" -eq 0 ]; then
        python render.py \
            -s "$source" -m "$model" --images images --iteration 7000 \
            --skip_test --voxel_size 0.00075 --depth_trunc 0.80 \
            --sdf_trunc 0.003 --num_cluster 20 \
            > "$model/render.log" 2>&1
        printf '%s\n' "$?" > "$model/render.exit"
    fi
    printf '%s\n' "$(( $(date +%s) - start ))" > "$model/wall_seconds.txt"
}

run_part gearboxopen
run_part gearboxclosed
printf 'complete\n' > "$ROOT/official_results/2dgs/all.done"
