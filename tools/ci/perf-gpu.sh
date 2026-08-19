#!/bin/bash

# The GPU half of the PR perftest comment: build master and this tree, measure.
set -euxo pipefail

# A card another process sits on, or a MIG slice, would mismeasure every case:
# say so before the builds, not ten minutes in.
tools/ci/gpu-exclusive.sh

rm -rf master
git clone --depth 1 https://github.com/flatironinstitute/finufft master
# One harness on both arms: the branch's whole perftest/.
rm -rf master/perftest && cp -r perftest master/perftest

for src in master .; do
	cmake -G Ninja -B "$src"/build "$src" -DFINUFFT_USE_CUDA=ON \
		-DFINUFFT_USE_CPU=OFF \
		-DFINUFFT_BUILD_TESTS=ON \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
		-DBUILD_TESTING=ON \
		-DFINUFFT_STATIC_LINKING=OFF
	cmake --build "$src"/build --target cuperftest -j "${PARALLEL:-8}"
done

tools/ci/gpu-exclusive.sh

uv run --script tools/perf-comment.py gpu \
	--master-cuperftest master/build/perftest/cuda/cuperftest \
	--pr-cuperftest build/perftest/cuda/cuperftest \
	--plot-output gpu_perf.svg --section-output gpu_perf.md
