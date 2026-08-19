#!/bin/bash

# One CPU FFT backend of the perftest report page, over every release tag.
set -euxo pipefail

# The node's, not the pod's quota: the record of what this shares.
grep -c ^processor /proc/cpuinfo
grep MemTotal /proc/meminfo

rm -rf ../builds && mkdir -p ../builds
for tag in $VERSIONS master; do
	if [ "$tag" = master ]; then src=.; else src=../$tag; fi
	# A release the harness can no longer build against drops out of the plots
	# rather than taking the page down with it. The runner names every tag it
	# skipped.
	cmake -G Ninja -B ../builds/"$tag" -S "$src" -DFINUFFT_BUILD_TESTS=ON \
		-DCMAKE_BUILD_TYPE=Release \
		-DFINUFFT_USE_DUCC0="$DUCC" &&
		cmake --build ../builds/"$tag" --target perftest -j "${PARALLEL:-8}" ||
		echo "skipping $tag: perftest did not build"
done

uv run --script perftest/run_perftest_ci.py \
	--backend "$BACKEND" \
	--builds-root ../builds \
	--tag-list "$VERSIONS master" \
	--cmake-cache-from master \
	--page-template docs/performance_section.rst.j2 \
	--output outputs
