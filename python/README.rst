Just scribbles for now, Barnett 3/27/17


Install prereqs:

sudo apt install python python-pip

sudo pip install pybind11


Linux build: (see ../makefile)

```
cd python
rm -rf build
python setup.py build_ext --inplace
```


Here's a couple of hints for the Mac OSX openmp install from Dan:

```
brew reinstall gcc --without-multilib
brew reinstall fftw --with-openmp
CC=/usr/local/bin/gcc-6 CXX=/usr/local/bin/g++-6
python setup.py build_ext --inplace
```


