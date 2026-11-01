#!/usr/bin/env bash
set -u

ROOT=/work/isaac_run
INPUT=$ROOT/input

run_one() {
    name="$1"
    mkdir -p "$ROOT/results/$name/reconstructions"
    python "$INPUT/reconstruct_classical.py" \
        --dataset "$ROOT/datasets/$name" \
        --out "$ROOT/results/$name/reconstructions" \
        --voxel-mm 0.75 --visual-hull-mm 1.25 \
        --poisson-depth 9 --nbv-views 24 \
        > "$ROOT/results/$name/classical.log" 2>&1
    printf '%s\n' "$?" > "$ROOT/results/$name/classical.exit"
}

run_one gearboxopen &
open_pid=$!
run_one gearboxclosed &
closed_pid=$!
wait "$open_pid"
wait "$closed_pid"
printf 'complete\n' > "$ROOT/results/classical.done"
