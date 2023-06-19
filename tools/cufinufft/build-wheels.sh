#!/bin/bash
set -e -u -x

function get_python_binary {
    version="$1"
    echo "/opt/python/$version/bin"
}

function repair_wheel {
    wheel="$1"
    if ! "${PYBIN}/auditwheel" show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        "${PYBIN}/auditwheel" repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}

# Explicitly list Python versions to build for
PYVERSIONS=(cp36-cp36m \
            cp37-cp37m \
            cp38-cp38 \
            cp39-cp39 \
            cp310-cp310 \
            cp311-cp311)

# Compile wheels
for PYVERSION in ${PYVERSIONS[@]}; do
    PYBIN=$(get_python_binary ${PYVERSION})

    "${PYBIN}/pip" install --upgrade pip
    "${PYBIN}/pip" install -r /io/python/cufinufft/requirements.txt
    "${PYBIN}/pip" install auditwheel pytest
    "${PYBIN}/pip" wheel /io/python/cufinufft --no-deps -w wheelhouse/
done


# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
for PYVERSION in ${PYVERSIONS[@]}; do
    PYBIN=$(get_python_binary ${PYVERSION})

    "${PYBIN}/pip" install cufinufft -f /io/wheelhouse
    "${PYBIN}/python" -m pytest /io/python/cufinufft/tests
done
