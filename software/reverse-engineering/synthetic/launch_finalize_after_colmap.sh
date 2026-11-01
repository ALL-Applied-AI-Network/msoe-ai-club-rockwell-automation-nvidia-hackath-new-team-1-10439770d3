#!/usr/bin/env bash
set -u

HOST_ROOT=/home/shadeform/work/isaac_run
CONTAINER_ROOT=/work/isaac_run

while [ ! -f "$HOST_ROOT/official_results/neuralangelo/all.done" ] \
   || [ ! -f "$HOST_ROOT/official_results/colmap_mvs/all.done" ] \
   || [ ! -f "$HOST_ROOT/official_results/2dgs/all.done" ] \
   || [ ! -f "$HOST_ROOT/results/compact_neural_sdf.done" ]; do
    sleep 10
done

if [ -f "$HOST_ROOT/results/aggregate/finalization.json" ]; then
    echo 'finalization already complete; preserving existing outputs'
    exit 0
fi

docker exec re2dgs118 bash "$CONTAINER_ROOT/input/run_finalization_in_container.sh"
