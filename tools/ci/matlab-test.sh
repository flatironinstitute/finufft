#!/bin/bash

# Build the MEX inside a stock MathWorks image and run the MATLAB tests.
# The image brings MATLAB only, so the toolchain comes from conda-forge
# into the workspace: the same compilers for every MATLAB release, no root.
# The GPU MEX is built only when the pod has a card and the image has the
# Parallel Computing Toolbox. Both tests call error() on a failure.
set -euxo pipefail

matlab_root=$(dirname "$(dirname "$(readlink -f "$(command -v matlab)")")")
gpu=OFF
if command -v nvidia-smi && [ -d "$matlab_root/toolbox/parallel" ]; then
	nvidia-smi
	gpu=ON
fi

mm="$WORKSPACE/.mm"
mkdir -p "$mm"
# Some images have no curl or wget; all have python3.
python3 -c 'import sys, urllib.request as u; u.urlretrieve(*sys.argv[1:])' \
	https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 "$mm/micromamba"
chmod +x "$mm/micromamba"
pkgs=(gxx_linux-64=13 cmake ninja git)
[ $gpu = ON ] && pkgs+=(cuda-nvcc=12.8 cuda-cudart-dev=12.8 libcufft-dev)
"$mm/micromamba" create -y -q -r "$mm/root" -p "$mm/env" -c conda-forge "${pkgs[@]}"
# The toolchain goes on PATH for the build only; MATLAB runs with its own
# libraries, not conda's.
tc="$mm/env/bin/x86_64-conda-linux-gnu"
PATH="$mm/env/bin:$PATH" CC=$tc-gcc CXX=$tc-g++ CUDAHOSTCXX=$tc-g++ cmake --preset matlab -B build \
	-DMatlab_ROOT_DIR="$matlab_root" \
	-DFINUFFT_USE_DUCC0=ON \
	-DFINUFFT_USE_CUDA=$gpu \
	${CUDA_ARCH:+-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH}
targets=(finufft_mex)
[ $gpu = ON ] && targets+=(cufinufft_mex)
PATH="$mm/env/bin:$PATH" cmake --build build -j "${PARALLEL:-8}" --target "${targets[@]}"

# A GPU leg must fail when MATLAB cannot use its card, else fullmathtest
# skips the GPU tests silently.
want=$([ $gpu = ON ] && echo "exist('cufinufft')==3 && canUseGPU()" || echo true)
card=$([ $gpu = ON ] && echo "gpuDevice().Name" || echo "'CPU only'")
matlab -batch "addpath(genpath('matlab')); addpath(genpath('build')); \
	ver; assert(exist('finufft')==3 && $want, 'MEX or GPU not usable'); \
	fullmathtest; tolsweeptest; \
	fprintf('SUMMARY %s %s PASS\n', version, $card)"
