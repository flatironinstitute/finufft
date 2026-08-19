#!/bin/bash

# Fail unless this pod holds a whole card with nothing else on it.
# A neighbour's kernels would land in the timings. nvidia.com/gpu is exclusive
# unless the cluster enables time-slicing or MPS, and this check catches that.
set -euxo pipefail

nvidia-smi --query-gpu=compute_mode,mig.mode.current --format=csv
nvidia-smi -L
# A slice is not a card: the L2, the TLB and the link stay shared with whoever
# holds the other slices, which a timing must not do.
# -L names a slice MIG, a whole card not.
if nvidia-smi -L | grep -q MIG; then
	echo "this pod holds a MIG slice, not a whole card"
	exit 1
fi
# A row the driver withholds reads "[Insufficient Permissions]", and that is not
# a neighbour: only a pid-shaped row is one.
apps=$(nvidia-smi --query-compute-apps=pid,used_gpu_memory \
	--format=csv,noheader | grep -E '^[0-9]+,' || true)
if [ -n "$apps" ]; then
	echo "another process is already on this GPU: $apps"
	exit 1
fi
