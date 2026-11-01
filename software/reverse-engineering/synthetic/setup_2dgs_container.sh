#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends git build-essential ninja-build ffmpeg libgl1 libglib2.0-0
python -m pip install --upgrade "pip<25" ninja cmake lit
python -m pip install \
    numpy==1.24.4 \
    pillow==10.2.0 open3d==0.18.0 mediapy==1.1.2 lpips==0.1.4 \
    scikit-image==0.21.0 tqdm==4.66.2 trimesh==4.3.2 plyfile \
    opencv-python-headless==4.10.0.84
cd /work/2d-gaussian-splatting
export TORCH_CUDA_ARCH_LIST=8.9
rm -rf submodules/simple-knn/build submodules/diff-surfel-rasterization/build
python -m pip install --no-build-isolation submodules/simple-knn
python -m pip install --no-build-isolation submodules/diff-surfel-rasterization
python - <<'PY'
import torch
import diff_surfel_rasterization
import simple_knn
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
PY
