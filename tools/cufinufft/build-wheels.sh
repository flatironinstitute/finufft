#!/bin/bash
set -e -u -x

function get_python_binary {
    version="$1"
    echo "/opt/python/$version/bin"
}

function repair_wheel {
    py_version="$1"
    wheel="$2"

    py_binary=$(get_python_binary "${py_version}")

    if ! "${py_binary}/pip" show auditwheel > /dev/null 2>&1; then
        "${py_binary}/pip" install auditwheel
    fi

    if ! "${py_binary}/auditwheel" show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        "${py_binary}/auditwheel" repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}

# Explicitly list Python versions to build for
py_versions=(cp36-cp36m \
            cp37-cp37m \
            cp38-cp38 \
            cp39-cp39 \
            cp310-cp310 \
            cp311-cp311 \
            cp312-cp312)

# NOTE: For CUDA 12, cp36-cp36m and cp37-cp37m are broken since these force an
# older version of pycuda (2022.1), which does not build under CUDA 12.

# Compile wheels
for py_version in ${py_versions[@]}; do
    py_binary=$(get_python_binary ${py_version})

    "${py_binary}/pip" install --upgrade pip
    "${py_binary}/pip" wheel /io/python/cufinufft --no-deps -w wheelhouse/
done


# Bundle external shared libraries into the wheels
audit_py_version="cp310-cp310"
for whl in wheelhouse/*.whl; do
    repair_wheel "$audit_py_version" "$whl"
done

# Install packages and test
for py_version in ${py_versions[@]}; do
    py_binary=$(get_python_binary ${py_version})

    "${py_binary}/pip" install --pre cufinufft -f /io/wheelhouse
    "${py_binary}/pip" install pytest
    "${py_binary}/pytest" /io/python/cufinufft/tests
done
