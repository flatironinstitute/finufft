#!/bin/bash -xe

# Helper Script For Building Wheels

manylinux_version=manylinux2014
cuda_version=11.2
dockerhub=janden

image_name=cufinufft-cuda${cuda_version}

echo "# Build the docker image"
docker build \
    --file tools/cufinufft/docker/cuda${cuda_version}/Dockerfile-x86_64 \
    --tag ${dockerhub}/cufinufft-cuda${cuda_version} \
    .

echo "# Create the container and start it"
docker create \
    --gpus all \
    --interactive \
    --tty \
    --volume $(pwd)/wheelhouse:/io/wheelhouse \
    --name ${image_name} \
    ${dockerhub}/${image_name}

docker start ${image_name}

echo "# Copy the code"
docker cp . ${image_name}:/io

echo "# Build the wheels"
docker exec ${image_name} \
    python3 -m pip wheel \
    --verbose \
    /io/python/cufinufft \
    --config-settings=cmake.define.FINUFFT_CUDA_ARCHITECTURES="50;60;70;80" \
    --config-settings=cmake.define.CMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets" \
    --config-settings=cmake.define.FINUFFT_ARCH_FLAGS="" \
    --config-settings=cmake.define.CMAKE_VERBOSE_MAKEFILE=ON \
    --no-deps \
    --wheel-dir /io/wheelhouse

wheel_name=$(docker exec ${image_name} bash -c 'ls /io/wheelhouse/cufinufft-*-linux_x86_64.whl')

echo "# Repair the wheels"
docker exec ${image_name} \
    python3 -m auditwheel repair \
    ${wheel_name} \
    --plat manylinux2014_x86_64 \
    --wheel-dir /io/wheelhouse/

echo "# Shut down the container and remove it"
docker stop ${image_name}
docker rm ${image_name}

echo "# Copy the wheels we care about to the dist folder"
mkdir -p dist
cp -v wheelhouse/cufinufft-*${manylinux_version}* dist

# TODO: Test installing the wheels and running pytest.
