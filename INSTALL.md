# Installation guide for FINUFFT

Barnett 3/24/17

## DEPENDENCIES

This library is only supported for unix/linux and Mac OSX right now.

For the basic libraries

- C++ compiler such as g++ packaged with GCC
- FFTW3
- GNU make

Optional:

- numdiff (preferred but not essential; enables pass-fail math validation)
- for Fortran wrappers: compiler such as gfortran
- for matlab/octave wrappers: matlab, or octave and its development libs
- for building new matlab/octave wrappers: mwrap

### Installing dependencies on various operating systems

On a Fedora/CentOS linux system, these dependencies can be installed as follows:

```
sudo yum install make gcc gcc-c++ gcc-gfortran fftw3 fftw3-devel libgomp octave octave-devel python python-pip
```
then see below for `numdiff` and `mwrap`.

then download the latest `numdiff` from http://gnu.mirrors.pair.com/savannah/savannah/numdiff/ and set it up via `./configure; make; sudo make install`

On Ubuntu linux:

```
sudo apt-get install make build-essential libfftw3-dev gfortran numdiff octave liboctave-dev
```

On Mac OSX:

Make sure you have `make` installed, eg via XCode.

Install gcc, for instance using pre-compiled binaries from
http://hpc.sourceforge.net/

Install homebrew from http://brew.sh

`brew install fftw`

Install `numdiff` as below.

Also see Mac OSX notes in the `python` directory.

Currently in Mac OSX, `make lib` fails to make the shared object library (.so);
however the static (.a) library is of reasonable size and works fine.


### Installing numdiff

[`numdiff`](http://www.nongnu.org/numdiff) by Ivano Primi extends `diff` to assess errors in floating-point outputs.
Download the latest `numdiff` from the above URL, un-tar the package, cd into it, then build via `./configure; make; sudo make install`

### Installing MWrap

[MWrap](http://www.cs.cornell.edu/~bindel/sw/mwrap)
is a very useful MEX interface generator by Dave Bindel.
Make sure you have `flex` and `bison` installed.
Download version 0.33 or later from http://www.cs.cornell.edu/~bindel/sw/mwrap, un-tar the package, cd into it, then
```
make
sudo cp mwrap /usr/local/bin/
```

## COMPILATION

Compile and test FINUFFT via

`make test`

or

`make test -j` to use all available cores.

This should compile the main libraries then run tests which should report zero crashes and zero fails. (If numdiff was not installed, it instead produces output that you will have to check by eye matches the requested accuracy.)
If you have an error then `cp makefile makefile.local`,
edit `makefile.local` to adjust compiler and other library options,
and use `make -f makefile.local test`.
Run `make` without arguments for full list of possible make tasks.

If there is an error in testing, consider filing a bug report; see [README](README.md)

### Building examples and wrappers

`make examples` to compile and run the examples for calling from C++ and from C.

The `examples` and `test` directories are good places to see usage examples.

`make fortran` to compile and run the fortran wrappers and examples.

`make matlab` to build the MEX interface to matlab.

`make octave` to build the MEX-like interface to octave.
