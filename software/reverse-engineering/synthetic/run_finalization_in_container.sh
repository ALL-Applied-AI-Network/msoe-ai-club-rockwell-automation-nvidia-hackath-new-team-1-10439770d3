#!/usr/bin/env bash
set -u

ROOT=/work/isaac_run
python "$ROOT/input/finalize_reverse_benchmark.py" \
    --root "$ROOT" --tools "$ROOT/input" \
    --pause-seconds 22957 --samples 120000 --turntable-points 160000 \
    > "$ROOT/results/finalization.log" 2>&1
status=$?
printf '%s\n' "$status" > "$ROOT/results/finalization.exit"
exit "$status"
