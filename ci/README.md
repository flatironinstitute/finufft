# Building Wheels

Noting how I built the wheels for myself in the future,
and in case it is helpful for when this is automated into the FI CI systems.
Typically these steps would be stages in CICD, and fully automated (up to the review).

```
# build the wheel
docker build -f ci/docker/cuda10.1/Dockerfile-x86_64 -t garrettwrong/cufinufft-1.0-manylinux2010 .

# optionally push it (might take a long time, because I didn't strip/clean the containers, tsk tsk)
docker push garrettwrong/cufinufft-1.0-manylinux2010

# Run the container, invoking the build-wheels script to generate the wheels
docker run --gpus all -it -v `pwd`/wheelhouse:/io/wheelhouse -e PLAT=manylinux2010_x86_64  garrettwrong/cufinufft-1.0-manylinux2010 /io/ci/build-wheels.sh

# Create a source distribution (requires you've built or have lib available)
python setup.py sdist

# Copy the wheels we care about to the dist folder
cp -v wheelhouse/cufinufft-1.0-cp3*manylinux2010* dist

# Push to Test PyPI for review/testing
twine upload -r testpypi dist/*

# Tag release.
## Can do in a repo and push or on manually on GH gui.

# Review wheels from test index
pip install -i https://test.pypi.org/simple/ --no-deps cufinufft

# Push to live index
## twine upload dist/*
```

Note that because the large size of the library, initially I expect this package will be over the 100MB limit imposed by PyPI.
Generally this just requires pushing a trivial source release, and then requesting an increase.
