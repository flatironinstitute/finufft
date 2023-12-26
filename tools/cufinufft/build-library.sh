#!/usr/bin/env bash
set -e -u -x

rm -rf /io/build
mkdir /io/build
cd /io/build

cmake -D FINUFFT_USE_CUDA=ON \
      -D FINUFFT_USE_CPU=OFF \
      -D FINUFFT_BUILD_TESTS=ON \
      -D CMAKE_CUDA_ARCHITECTURES="50;60;70;80" \
      -D CMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets" \
      -D BUILD_TESTING=ON \
      ..

make -j4
