#!/usr/bin/env bash

set -e -x

# Move pyproject.toml to root (otherwise no way to include C++ sources in sdist).
mv finufft/python/finufft/pyproject.toml finufft/

# Fix paths in pyproject.toml to reflect the new directory structure.
toml set --toml-path finufft/pyproject.toml \
         tool.scikit-build.cmake.source-dir "."
toml set --toml-path finufft/pyproject.toml \
         tool.scikit-build.wheel.packages --to-array "[\"python/finufft/finufft\"]"
toml set --toml-path finufft/pyproject.toml \
         tool.scikit-build.metadata.version.input "python/finufft/finufft/__init__.py"

# Package the sdist.
python3 -m build --verbose --sdist --outdir wheelhouse finufft
