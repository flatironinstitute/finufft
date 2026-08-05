#!/usr/bin/env bash
# Build a manylinux cufinufft wheel. Run from the repository root inside
# tools/cufinufft/docker/Dockerfile-x86_64:
#   build-wheel.sh "<cuda-archs>" [outdir]     e.g. build-wheel.sh "70;80;90"
set -eux

archs=${1:?cuda architectures, e.g. "70;80"}
out=${2:-wheelhouse}

python3 -m pip wheel python/cufinufft --no-deps -w "${out}/raw" \
	--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES="${archs}" \
	--config-settings=cmake.define.CMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets"

# As in the released wheels ([tool.cibuildwheel] in python/cufinufft/pyproject.toml)
# the CUDA runtime and cuFFT stay unbundled - they are the user's - but matched by
# soname glob so this works on any CUDA version. libcuda.so.1 is the host driver's.
python3 -m auditwheel repair "${out}"/raw/cufinufft-*.whl \
	--exclude 'libcudart.so.*' \
	--exclude 'libcufft.so.*' \
	--exclude libcuda.so.1 \
	--plat manylinux_2_28_x86_64 \
	-w "${out}"
