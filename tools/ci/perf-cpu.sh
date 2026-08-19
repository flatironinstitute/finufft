#!/bin/bash

# The CPU half of the PR perftest comment: build master and this tree, measure.
set -euxo pipefail

# The node's, not the pod's quota: the record of what this shares.
grep -c ^processor /proc/cpuinfo
grep MemTotal /proc/meminfo
# The runner pins each single-core case to a core of its own with taskset, and
# spreads a whole-node case over the memory controllers with set_mempolicy,
# which is a syscall rather than a binary this image would have to carry.
command -v taskset
cat /sys/devices/system/node/online
# The pod's cpuset against the node's NUMA layout: an exclusive pod gets whole
# cores, but a count below the node's splits them unevenly over the sockets.
grep Cpus_allowed_list /proc/self/status
grep . /sys/devices/system/node/node*/cpulist

rm -rf master
git clone --depth 1 https://github.com/flatironinstitute/finufft master
# One harness on both arms: the branch's whole perftest/, as on the GPU half.
rm -rf master/perftest && cp -r perftest master/perftest

# The comment measures DUCC0 only; the performance page runs both backends,
# so an FFTW-side retune still shows up there on every master push.
for src in master .; do
	cmake -G Ninja -B "$src"/build-DUCC0 "$src" -DFINUFFT_USE_CPU=ON \
		-DFINUFFT_USE_CUDA=OFF \
		-DFINUFFT_USE_DUCC0=ON \
		-DFINUFFT_BUILD_TESTS=ON \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_TESTING=ON
	cmake --build "$src"/build-DUCC0 --target perftest -j "${PARALLEL:-8}"
done

# This tree's runner, as it is this tree's harness on both arms: the plumbing
# rides the branch it measures.
uv run --script tools/perf-comment.py cpu \
	--master-perftest DUCC0=master/build-DUCC0/perftest/perftest \
	--pr-perftest DUCC0=build-DUCC0/perftest/perftest \
	--plot-output cpu_perf.svg --section-output cpu_perf.md
