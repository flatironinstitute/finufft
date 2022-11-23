#!/bin/bash

set -e -x

# Replace native compilation flags with more generic ones.
cp make.inc.manylinux make.inc

# Clean up the build and make the library.
make clean
make lib

# Test to make sure everything is ok.
make test

# Needed for pip install to work
export FINUFFT_DIR=$(pwd)
# Needed for auditwheel to find the dynamic libraries
export LD_LIBRARY_PATH=${FINUFFT_DIR}/lib:${LD_LIBRARY_PATH}

# Explicitly list Python versions to build
versions=("cp36-cp36m"
          "cp37-cp37m"
          "cp38-cp38"
          "cp39-cp39"
          "cp310-cp310"
          "cp311-cp311"
          "pp37-pypy37_pp73"
          "pp38-pypy38_pp73")

pys=()
for version in "${versions[@]}"; do
    pys+=("/opt/python/${version}/bin")
done

# build wheel
for pybin in "${pys[@]}"; do
    "${pybin}/pip" install --upgrade pip
    "${pybin}/pip" install auditwheel wheel twine numpy
    "${pybin}/pip" wheel ./python -w python/wheelhouse
done

# fix wheel
for whl in python/wheelhouse/finufft-*.whl; do
    auditwheel repair "$whl" -w python/wheelhouse/
done

# test wheel
for pybin in "${pys[@]}"; do
    "${pybin}/pip" install finufft -f ./python/wheelhouse/
    "${pybin}/python" ./python/test/run_accuracy_tests.py
    "${pybin}/python" ./python/examples/simple1d1.py
done
