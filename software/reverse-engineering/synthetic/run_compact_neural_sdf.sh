#!/usr/bin/env bash
set -u

ROOT=/work/isaac_run
INPUT="$ROOT/input"

run_one() {
    name="$1"
    out="$ROOT/results/$name/reconstructions"
    log="$ROOT/results/$name/neural_sdf_depth.log"
    mkdir -p "$out"
    if [ -f "$out/neural_sdf_depth.json" ]; then
        echo "[compact-neural-sdf] $name already has a result; preserving it" >> "$log"
        return 0
    fi
    python "$INPUT/reconstruct_neural_sdf.py" \
        --dataset "$ROOT/datasets/$name" \
        --out "$out" \
        --steps 1100 --batch 32768 \
        --surface-voxel-mm 0.70 --grid-mm 0.85 \
        > "$log" 2>&1
    printf '%s\n' "$?" > "$ROOT/results/$name/neural_sdf_depth.exit"
}

rm -f "$ROOT/results/compact_neural_sdf.done"
run_one gearboxopen
run_one gearboxclosed
printf 'complete\n' > "$ROOT/results/compact_neural_sdf.done"
