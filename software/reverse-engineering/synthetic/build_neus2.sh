#!/usr/bin/env bash
set -euo pipefail
cd /work/NeuS2
rm -rf build
cmake . -B build -DNGP_BUILD_WITH_GUI=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --config RelWithDebInfo -j 12
