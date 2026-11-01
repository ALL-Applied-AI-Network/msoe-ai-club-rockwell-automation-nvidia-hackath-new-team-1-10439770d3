#!/usr/bin/env bash
set -u

ROOT=/workspace/reverse_engineering_benchmark
INPUT=$ROOT/input

run_capture() {
    name="$1"
    source="$2"
    output="$ROOT/datasets/$name"
    mkdir -p "$output"
    /isaac-sim/python.sh "$INPUT/capture_isaac_orbit.py" \
        --stl "$source" \
        --out "$output" \
        --width 960 --height 720 \
        --azimuths 18 \
        --elevations 18 32 46 60 74 \
        --radius-m 0.42 \
        --photometric-keyframes 12 \
        --dual-pass \
        > "$output/capture.log" 2>&1
    code=$?
    printf '%s\n' "$code" > "$output/capture.exit"
    return "$code"
}

mkdir -p "$ROOT/datasets"
run_capture gearboxopen "$INPUT/gearboxopen.stl"
open_code=$?
run_capture gearboxclosed "$INPUT/gearboxclosed.stl"
closed_code=$?
printf 'open=%s closed=%s\n' "$open_code" "$closed_code" > "$ROOT/datasets/all_capture.exit"
exit $((open_code != 0 || closed_code != 0))
