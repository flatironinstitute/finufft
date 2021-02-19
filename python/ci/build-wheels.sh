#!/bin/bash

set -e -x

cd /io/

# Replace native compilation flags with more generic ones.
cp make.inc.manylinux make.inc

# Clean up the build and make the library.
make clean
make lib

# Test to make sure everything is ok.
make test

# Remove make.inc now that we're done.
rm make.inc

# Needed for pip install to work
export FINUFFT_DIR=$(pwd)
# Needed for auditwheel to find the dynamic libraries
export LD_LIBRARY_PATH=${FINUFFT_DIR}/lib:${LD_LIBRARY_PATH}

pys=(/opt/python/*/bin)

# Filter out old Python versions
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
