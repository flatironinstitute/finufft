#!/bin/bash
set -e -u -x

py_versions=(cp36-cp36m \
            cp37-cp37m \
            cp38-cp38 \
            cp39-cp39 \
            cp310-cp310 \
            cp311-cp311)

for py_version in ${py_versions[@]}; do
    py_binary="/opt/python/$py_version/bin"

    "${py_binary}/pip" install --upgrade pip

    "${py_binary}/pip" install /io/python/cufinufft

    "${py_binary}/pip" install pytest
    "${py_binary}/pytest" /io/python/cufinufft/tests
done
