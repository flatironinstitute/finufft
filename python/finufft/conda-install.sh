#!/usr/bin/env bash
# Build and test FINUFFT's Python package from this checkout in a conda environment.
# Run from anywhere: `bash python/finufft/conda-install.sh`.
set -euo pipefail
cd "$(dirname "$0")"
eval "$(conda shell.bash hook)"

# sphinx tag (don't remove): @conda_finufft_start
conda env create -f environment.yml
conda activate finufft-build
pip install . pytest
pytest test
# pytest does not run examples/; each example asserts its own math check.
for example in examples/*.py; do python "$example"; done
# sphinx tag (don't remove): @conda_finufft_end
