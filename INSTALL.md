# Installation notes for FINUFFT

Barnett 3/21/17

### DEPENDENCIES

For the basic libraries

- C++ compiler such as g++
- FFTW3
- GNU make
- numdiff (optional)
- Optionally, OpenMP (however, the makefile can be adjusted for single-threaded operation)

For the Fortran wrappers

- Fortran compiler such as gfortran (see settings in the makefile)

### INSTALLATION ON DIFFERENT OPERATING SYSTEMS

On a Fedora/CentOS linux system, these dependencies can be installed as follows:

`sudo yum install make gcc gcc-c++ gcc-gfortran fftw3 fftw3-devel libgomp`

then download the latest `numdiff` from http://gnu.mirrors.pair.com/savannah/savannah/numdiff/ and set it up via `./configure; make; sudo make install`

On Ubuntu linux:

sudo apt-get install make build-essential libfftw3-dev gfortran numdiff

On Mac OSX:

Install gcc, for instance using pre-compiled binaries from
http://hpc.sourceforge.net/

Install homebrew from http://brew/sh

`brew install fftw`

In the makefile for FINUFFT, uncomment the line for Mac OSX.


### COMPILATION

Compile the library using

make test

This should compile the main libraries then run tests.
If you have an error then `cp makefile makefile.local`,
edit `makefile.local` to adjust compiler and other library options,
and use `make -f makefile.local test`.

Here are some other make tasks (see `make` without arguments for full list):

- `make examples` : compile the demos in `examples/`
- `make test` : mathematical validation of the library and components
- `make perftest` : multi-threaded and single-threaded performance tests
- `make fortran` : compile and test the fortran interfaces  

Linking to the library: to link to the static library
`lib/libfinufft.a` use the compiler flag `-Llib/libfinufft.a` (or
replacing this by the absolute location of this library). In your
C/C\++ code you will need to include the header `src/finufft.h`.
You may also try linking to the shared object `lib/libfinufft.so`;
however this is currently experimental.
