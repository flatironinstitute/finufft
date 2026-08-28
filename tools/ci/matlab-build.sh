#!/bin/bash

# Build the MATLAB MEX interface and package it exactly as a user receives it.
#
# No MATLAB session here: this half only compiles. matlab-consume.sh runs the
# package, on Linux in an image that has MATLAB and no toolchain, so a MEX that
# only works beside its own build tree cannot pass.
#
# FINUFFT_CI_GPU=1 also builds cufinufft_mex, which needs a card only at test
# time, not here.
set -euxo pipefail

gpu=${FINUFFT_CI_GPU:-0}
cuda=OFF
[[ "$gpu" == "1" ]] && cuda=ON

cmake --preset matlab -B build \
	-DFINUFFT_USE_CUDA=$cuda \
	${CUDA_ARCH:+-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH}
cmake --build build --target finufft_mex -j "${PARALLEL:-8}"
if [[ "$gpu" == "1" ]]; then
	cmake --build build --target cufinufft_mex -j "${PARALLEL:-8}"
fi

# CPack, not `cmake --install`: the archive is the artifact the MATLAB users get,
# and it carries the wrappers and the MEX with no CMake package around them.
cpack --config build/CPackConfig.cmake -B "$PWD"
ls -l finufft-matlab-mex-*
