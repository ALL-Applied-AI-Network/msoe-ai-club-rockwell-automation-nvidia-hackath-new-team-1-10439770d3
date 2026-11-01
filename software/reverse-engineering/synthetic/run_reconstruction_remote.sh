#!/usr/bin/env bash
set -u

ROOT=/workspace/reverse_engineering_benchmark
INPUT=$ROOT/input

run_classical() {
    name="$1"
    dataset="$ROOT/datasets/$name"
    output="$ROOT/results/$name/reconstructions"
    mkdir -p "$output"
    /isaac-sim/python.sh "$INPUT/reconstruct_classical.py" \
        --dataset "$dataset" --out "$output" \
        --voxel-mm 0.75 --visual-hull-mm 1.25 --poisson-depth 9 --nbv-views 24 \
        > "$ROOT/results/$name/classical.log" 2>&1
    printf '%s\n' "$?" > "$ROOT/results/$name/classical.exit"
}

run_neural() {
    name="$1"
    dataset="$ROOT/datasets/$name"
    output="$ROOT/results/$name/reconstructions"
    /isaac-sim/python.sh "$INPUT/reconstruct_neural_sdf.py" \
        --dataset "$dataset" --out "$output" \
        --steps 1100 --batch 32768 --surface-voxel-mm 0.70 --grid-mm 0.85 \
        > "$ROOT/results/$name/neural.log" 2>&1
    printf '%s\n' "$?" > "$ROOT/results/$name/neural.exit"
}

run_eval() {
    name="$1"
    dataset="$ROOT/datasets/$name"
    recon="$ROOT/results/$name/reconstructions"
    evaluation="$ROOT/results/$name/evaluation"
    mkdir -p "$evaluation"
    /isaac-sim/python.sh "$INPUT/evaluate_reconstructions.py" \
        --dataset "$dataset" --reconstructions "$recon" --out "$evaluation" --samples 120000 \
        > "$ROOT/results/$name/evaluation.log" 2>&1
    printf '%s\n' "$?" > "$ROOT/results/$name/evaluation.exit"
}

mkdir -p "$ROOT/results/gearboxopen" "$ROOT/results/gearboxclosed"
run_classical gearboxopen &
open_pid=$!
run_classical gearboxclosed &
closed_pid=$!
wait "$open_pid"
wait "$closed_pid"

# Neural training is serialized to keep GPU timing and memory comparable.
run_neural gearboxopen
run_neural gearboxclosed
run_eval gearboxopen &
eval_open_pid=$!
run_eval gearboxclosed &
eval_closed_pid=$!
wait "$eval_open_pid"
wait "$eval_closed_pid"

printf 'complete\n' > "$ROOT/results/all.done"
