#!/usr/bin/env bash
set -u

ROOT=/work/isaac_run
REPO=/workspace/neuralangelo

run_part() {
    name="$1"
    short="${name#gearbox}"
    data="$ROOT/official_inputs/neural/$name"
    result="$ROOT/official_results/neuralangelo/$name"
    config="$REPO/projects/neuralangelo/configs/custom/benchmark_${short}.yaml"
    mkdir -p "$result"
    cd "$REPO"
    python projects/neuralangelo/scripts/generate_config.py \
        --sequence_name "benchmark_${short}" --data_dir "$data" \
        --scene_type object --val_short_size 240 \
        > "$result/config.log" 2>&1
    start=$(date +%s)
    python train.py --single_gpu --config "$config" --logdir "$result/logs" \
        --show_pbar --max_iter=3500 --validation_iter=3500 \
        --checkpoint.save_iter=3500 \
        --model.object.sdf.encoding.coarse2fine.step=1500 \
        --optim.sched.warm_up_end=1500 \
        --optim.sched.two_steps='[2200,3000]' \
        > "$result/train.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$result/train.exit"
    printf '%s\n' "$(( $(date +%s) - start ))" > "$result/train_wall_seconds.txt"
    if [ "$status" -ne 0 ]; then
        return
    fi
    checkpoint=$(find "$result/logs" -type f -name '*.pt' | sort | tail -n 1)
    saved_config="$result/logs/config.yaml"
    if [ ! -f "$saved_config" ]; then
        saved_config="$config"
    fi
    python projects/neuralangelo/scripts/extract_mesh.py --single_gpu \
        --config "$saved_config" --checkpoint "$checkpoint" \
        --output_file "$result/neuralangelo.ply" \
        --resolution 512 --block_res 128 \
        > "$result/mesh.log" 2>&1
    printf '%s\n' "$?" > "$result/mesh.exit"
}

run_part gearboxopen
run_part gearboxclosed
printf 'complete\n' > "$ROOT/official_results/neuralangelo/all.done"
