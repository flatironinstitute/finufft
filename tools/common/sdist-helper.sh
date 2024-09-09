#!/usr/bin/env bash

set -e -x

dockerhub=janden
image=finufft-sdist

docker build --file tools/common/docker/Dockerfile-x86_64 \
             --tag ${dockerhub}/${image} \
             .

docker run --volume $(pwd)/wheelhouse:/io/wheelhouse \
           ${dockerhub}/${image} \
           /io/finufft/tools/finufft/build-sdist.sh

docker run --volume $(pwd)/wheelhouse:/io/wheelhouse \
           ${dockerhub}/${image} \
           /io/finufft/tools/cufinufft/build-sdist.sh
