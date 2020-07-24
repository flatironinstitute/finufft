#!/bin/bash

# Copyright (c) 2019, Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/azure-wheel-helpers for details.

# Based on https://github.com/pypa/python-manylinux-demo/blob/master/travis/build-wheels.sh
# with CC0 license here: https://github.com/pypa/python-manylinux-demo/blob/master/LICENSE

set -e -x

curl http://www.fftw.org/fftw-3.3.8.tar.gz --output fftw-3.3.8.tar.gz
tar -xvzf fftw-3.3.8.tar.gz
cd fftw-3.3.8
CFLAGS=-fPIC ./configure --enable-threads --enable-openmp
make
make install
make clean
CFLAGS=-fPIC ./configure --enable-threads --enable-openmp --enable-float
make
make install

cd /io/
export FINUFFT_DIR=$(pwd)
make lib
make test

pys=(/opt/python/*/bin)

# Filter out Python 3.4
pys=(${pys[@]//*34*/})

for PYBIN in "${pys[@]}"; do
    "${PYBIN}/pip" install auditwheel wheel twine pybind11 python-dotenv numpy
    "${PYBIN}/pip" wheel /io/python -w python/wheelhouse    
done

for whl in python/wheelhouse/$package_name-*.whl; do
    auditwheel repair "$whl" -w /io/python/wheelhouse/
done
