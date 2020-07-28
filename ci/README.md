# Building Wheels

Noting how I built the wheels for myself in the future,
and in case it is helpful for when this is automated into the FI CI systems

```
# build the wheel
docker build -f ci/docker/cuda10.1/Dockerfile-x86_64 -t garrettwrong/cufinufft-1.0-manylinux2010 .

# optionally push it (might take a long time, because I didn't strip/clean the containers, tsk tsk)
docker push garrettwrong/cufinufft-1.0-manylinux2010

# Run the container, invoking the build-wheels script to generate the wheels
docker run --gpus all -it -v `pwd`/wheelhouse:/io/wheelhouse -e PLAT=manylinux2010_x86_64 test_cufinufft_manylinux2010 /io/ci/build-wheels.sh

# Push to Test PyPI for review/testing


```