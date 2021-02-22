#!/bin/bash -xe

# Helper Script For Building Wheels

cufinufft_version=1.2
manylinux_version=manylinux2014
cuda_version=11.0
dockerhub=garrettwrong


echo "# build the wheel"
docker build -f ci/docker/cuda${cuda_version}/Dockerfile-x86_64 -t ${dockerhub}/cufinufft-${cufinufft_version}-${manylinux_version} .


echo "# Run the container, invoking the build-wheels script to generate the wheels"
docker run --gpus all -it -v `pwd`/wheelhouse:/io/wheelhouse -e PLAT=${manylinux_version}_x86_64  ${dockerhub}/cufinufft-${cufinufft_version}-${manylinux_version} /io/ci/build-wheels.sh

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
