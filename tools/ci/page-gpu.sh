#!/bin/bash

# The GPU section of the perftest report page, over every release tag.
set -euxo pipefail

tools/ci/gpu-exclusive.sh

# CUDA 12.8 dropped the CUDA::nvToolsExt target that releases before 2.4 link.
# An empty stand-in keeps their CMake resolving, and NVTX only traces, so no
# timing on this page depends on it.
cat >"$PWD/nvtx-shim.cmake" <<'CMAKE'
if(NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt INTERFACE IMPORTED GLOBAL)
endif()
CMAKE

rm -rf ../builds && mkdir -p ../builds
for tag in $VERSIONS master; do
	if [ "$tag" = master ]; then src=.; else src=../$tag; fi
	# A release that predates what a case asks of it - type 3, or an opts field -
	# drops out of that case's plot. The runner names every tag it skipped, at
	# build time and at run time both.
	cmake -G Ninja -B ../builds/"$tag" -S "$src" -DFINUFFT_USE_CUDA=ON \
		-DFINUFFT_USE_CPU=OFF \
		-DFINUFFT_BUILD_TESTS=ON \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
		-DBUILD_TESTING=ON \
		-DCMAKE_PROJECT_INCLUDE="$PWD/nvtx-shim.cmake" \
		-DFINUFFT_STATIC_LINKING=OFF &&
		cmake --build ../builds/"$tag" --target cuperftest -j "${PARALLEL:-8}" ||
		echo "skipping $tag: cuperftest did not build"
done

uv run --script perftest/run_perftest_ci.py \
	--backend cuda \
	--builds-root ../builds \
	--tag-list "$VERSIONS master" \
	--cmake-cache-from master \
	--page-template docs/performance_section.rst.j2 \
	--output outputs
