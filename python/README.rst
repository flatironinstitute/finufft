

Linux build:

```
cd python
rm -rf build
python setup.py build_ext --inplace
```


Here's a couple of hints for the Mac OSX install from Dan:

```
brew reinstall gcc --without-multilib
brew reinstall fftw --with-openmp
CC=/usr/local/bin/gcc-6 CXX=/usr/local/bin/g++-6
python setup.py build_ext --inplace
```


