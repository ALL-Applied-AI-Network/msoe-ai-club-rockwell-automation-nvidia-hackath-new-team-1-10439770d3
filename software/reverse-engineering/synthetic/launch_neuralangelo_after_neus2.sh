#!/usr/bin/env bash
set -u
until [ -f /home/shadeform/work/isaac_run/official_results/neus2/all.done ]; do
    sleep 10
done
docker run --rm --name neuralangelo-benchmark --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /home/shadeform/work/neuralangelo:/workspace/neuralangelo \
    -v /home/shadeform/work/isaac_run:/work/isaac_run \
    -w /workspace/neuralangelo \
    chenhsuanlin/neuralangelo:23.04-py3 \
    bash /work/isaac_run/input/run_official_neuralangelo.sh \
    > /tmp/neuralangelo_container.log 2>&1
printf '%s\n' "$?" > /tmp/neuralangelo_container.exit
