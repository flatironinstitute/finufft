#!/usr/bin/env bash
# Build and test cuFINUFFT's Python package from this checkout in a conda environment.
# Run from anywhere: `bash python/cufinufft/conda-install.sh`.
# No -u: conda's cuda-nvcc activation script reads the unset NVCC_PREPEND_FLAGS.
set -eo pipefail
cd "$(dirname "$0")"
eval "$(conda shell.bash hook)"

# sphinx tag (don't remove): @conda_cufinufft_start
conda env create -f environment.yml
conda activate cufinufft-build
pip install . pytest
# tests/test_examples.py also runs every examples/ script for the given framework.
pytest --framework=cupy tests
# sphinx tag (don't remove): @conda_cufinufft_end
