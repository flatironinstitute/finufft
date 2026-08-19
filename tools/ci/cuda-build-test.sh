#!/bin/bash

# Build cufinufft for the card this pod holds, then run its tests.
# CUDA_ARCH comes from the card itself, so no list of architectures can go
# stale against the hardware.
set -euxo pipefail

nvidia-smi
nvcc --version
g++ --version
cmake -G Ninja -B build . -DFINUFFT_USE_CUDA=ON \
	-DFINUFFT_USE_CPU=OFF \
	-DFINUFFT_BUILD_TESTS=ON \
	-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
	-DBUILD_TESTING=ON \
	-DFINUFFT_STATIC_LINKING=OFF
cmake --build build -j "${PARALLEL:-8}"
ctest --test-dir build/test/cuda --output-on-failure
