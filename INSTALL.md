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

### INSTALLATION ON VARIOUS OPERATING SYSTEMS

On a Fedora/CentOS linux system, these dependencies can be installed as follows:

`sudo yum install make gcc gcc-c++ gcc-gfortran fftw3 fftw3-devel libgomp`

then download the latest `numdiff` from http://gnu.mirrors.pair.com/savannah/savannah/numdiff/ and set it up via `./configure; make; sudo make install`

On Ubuntu linux:

sudo apt-get install make build-essential libfftw3-dev gfortran numdiff

On Mac OSX:

Make sure you have `make` installed, eg via XCode.

Install gcc, for instance using pre-compiled binaries from
http://hpc.sourceforge.net/

Install homebrew from http://brew/sh

`brew install fftw`

Download the latest `numdiff` from http://gnu.mirrors.pair.com/savannah/savannah/numdiff/ and set it up via `./configure; make; sudo make install`

Currently for Mac OSX, `make lib` fails to make the shared object library (.so)

### COMPILATION

Compile the library using

`make test`

This should compile the main libraries then run tests.
If you have an error then `cp makefile makefile.local`,
edit `makefile.local` to adjust compiler and other library options,
and use `make -f makefile.local test`.
Run `make` without arguments for full list of possible make tasks.
