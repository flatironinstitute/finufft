#!/bin/bash

# Package with the release tooling (auditwheel, manylinux_2_28) and install
# that wheel, not the build tree. Only the pod's own architecture is compiled,
# unlike the released all-arch wheels.
set -euxo pipefail

# The image already has auditwheel and the GPU frameworks; the venv is only so
# the wheel installs somewhere writable.
python3 -m venv --system-site-packages "$HOME/venv"
# activate reads variables it does not set, which set -u would call an error.
set +u && source "$HOME/venv/bin/activate" && set -u
tools/cufinufft/build-wheel.sh "$CUDA_ARCH" "wheelhouse/cuda$CUDA"
python3 -m pip install --no-cache-dir wheelhouse/cuda"$CUDA"/cufinufft-*manylinux*.whl
python3 -c "import cufinufft; print(cufinufft.__version__)"
