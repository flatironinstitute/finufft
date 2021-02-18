#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! "${PYBIN}/auditwheel" show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        "${PYBIN}/auditwheel" repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}


# Compile wheels
for PYBIN in /opt/python/cp3[6789]*/bin; do
    "${PYBIN}/pip" install --upgrade pip
    "${PYBIN}/pip" install -r /io/python/cufinufft/requirements.txt
    "${PYBIN}/pip" install auditwheel pytest
    "${PYBIN}/pip" wheel /io/ --no-deps -w wheelhouse/
done


# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done


# Install packages and test
for PYBIN in /opt/python/cp3[6789]*/bin/; do
    "${PYBIN}/pip" install cufinufft -f /io/wheelhouse
    "${PYBIN}/python" -m pytest /io/python/cufinufft/tests
done
