#!/bin/bash -xe

# Helper Script For Building Wheels

cufinufft_version=2.2
manylinux_version=manylinux2014
cuda_version=11.0
dockerhub=janden


echo "# Build the docker image"
docker build \
    --file tools/cufinufft/docker/cuda${cuda_version}/Dockerfile-x86_64 \
    --tag ${dockerhub}/cufinufft-${cufinufft_version}-${manylinux_version} .


echo "# Create the container and start it"
docker create \
    --gpus all \
    --interactive \
    --tty \
    --volume $(pwd)/wheelhouse:/io/wheelhouse \
    --env PLAT=${manylinux_version}_x86_64 \
    --env LIBRARY_PATH="/io/build" \
    --env LD_LIBRARY_PATH="/io/build" \
    --name cufinufft \
    ${dockerhub}/cufinufft-${cufinufft_version}-${manylinux_version}

docker start cufinufft

echo "# Copy the code and build the library"
docker cp . cufinufft:/io
docker exec cufinufft /io/tools/cufinufft/build-library.sh

echo "# Build the wheels"
docker exec cufinufft /io/tools/cufinufft/build-wheels.sh

echo "# Shut down the container and remove it"
docker stop cufinufft
docker rm cufinufft

echo "# Copy the wheels we care about to the dist folder"
mkdir -p dist
cp -v wheelhouse/cufinufft-${cufinufft_version}-cp3*${manylinux_version}* dist

echo "The following steps should be performed manually for now.\n"

echo "# Push to Test PyPI for review/testing"
echo "#twine upload -r testpypi dist/*"
echo


echo "# Tag release."
## Can do in a repo and push or on manually on GH gui.
echo


echo "# Review wheels from test index"
echo "#pip install -i https://test.pypi.org/simple/ --no-deps cufinufft"
echo


echo "# Push to live index"
echo "## twine upload dist/*"
echo


echo "# optionally push it (might take a long time)."
echo "#docker push ${dockerhub}/cufinufft-${cufinufft_version}-${manylinux_version}"
