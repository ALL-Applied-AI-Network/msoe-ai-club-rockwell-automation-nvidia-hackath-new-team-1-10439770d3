#!/usr/bin/env bash
set -u
until [ -f /home/shadeform/work/isaac_run/official_results/neuralangelo/all.done ]; do
    sleep 10
done
docker run --rm --name colmap-benchmark --gpus all --ipc=host \
    -v /home/shadeform/work/isaac_run:/work/isaac_run \
    chenhsuanlin/colmap:3.8 \
    bash /work/isaac_run/input/run_official_colmap.sh \
    > /tmp/colmap_benchmark.log 2>&1
