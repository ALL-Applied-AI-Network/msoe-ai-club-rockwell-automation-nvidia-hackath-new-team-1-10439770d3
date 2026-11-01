#!/usr/bin/env bash
set -u
until [ -f /work/isaac_run/official_results/2dgs/all.done ]; do
    sleep 10
done
bash /work/isaac_run/input/run_official_neus2.sh
