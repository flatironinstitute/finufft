#!/bin/bash

# Copyright (c) 2019, Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/azure-wheel-helpers for details.

# Based on https://github.com/pypa/python-manylinux-demo/blob/master/travis/build-wheels.sh
# with CC0 license here: https://github.com/pypa/python-manylinux-demo/blob/master/LICENSE

set -e -x

cd /io/
make lib
make test

# Needed for pip install to work
export FINUFFT_DIR=$(pwd)
# Needed for auditwheel to find the dynamic libraries
export LD_LIBRARY_PATH=${FINUFFT_DIR}/lib:${LD_LIBRARY_PATH}

pys=(/opt/python/*/bin)

# Filter out Python 3.4
pys=(${pys[@]//*27*/})
pys=(${pys[@]//*34*/})
pys=(${pys[@]//*35*/})

for PYBIN in "${pys[@]}"; do
    "${PYBIN}/pip" install auditwheel wheel twine numpy
    "${PYBIN}/pip" wheel /io/python -w python/wheelhouse    
done

for whl in python/wheelhouse/$package_name-*.whl; do
    auditwheel repair "$whl" -w /io/python/wheelhouse/
done
