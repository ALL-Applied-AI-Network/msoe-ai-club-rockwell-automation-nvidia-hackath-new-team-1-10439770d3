#!/usr/bin/env bash
set -u

ROOT=/work/isaac_run
REPO=/work/NeuS2

run_part() {
    name="$1"
    scene="$ROOT/official_inputs/neural/$name/transforms.json"
    result="$ROOT/official_results/neus2/$name"
    mkdir -p "$result"
    cd "$REPO"
    start=$(date +%s)
    PYTHONPATH="$REPO/build:${PYTHONPATH:-}" python scripts/run_mesh_only.py \
        --scene "$scene" --name "benchmark_$name" --network dtu.json \
        --n_steps 10000 --marching_cubes_res 384 \
        > "$result/train.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$result/train.exit"
    printf '%s\n' "$(( $(date +%s) - start ))" > "$result/wall_seconds.txt"
    mesh="$REPO/output/benchmark_$name/mesh/10000.obj"
    if [ -f "$mesh" ]; then
        cp "$mesh" "$result/neus2_raw.obj"
    fi
}

run_part gearboxopen
run_part gearboxclosed
printf 'complete\n' > "$ROOT/official_results/neus2/all.done"
