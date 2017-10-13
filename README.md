# Flatiron Institute Nonuniform Fast Fourier Transform libraries: FINUFFT

Version 0.95  (10/2/2017)

### Alex H. Barnett and Jeremy F. Magland

### Purpose

This is a lightweight library to compute the nonuniform FFT to a specified precision, in one, two, or three dimensions.
This task is to approximate various exponential sums involving large numbers of terms and output indices, in close to linear time.
The speedup over naive evaluation of the sums is similar to that achieved by the FFT. For instance, for _N_ terms and _N_ output indices, the computation time is _O_(_N_ log _N_) as opposed to the naive _O_(_N_<sup>2</sup>).
For convenience, we conform to the simple existing interfaces of the
[CMCL NUFFT libraries of Greengard--Lee from 2004](http://www.cims.nyu.edu/cmcl/nufft/nufft.html), apart from the normalization factor of type-1.
Our main innovations are: speed (enhanced by a new functional form for the spreading kernel), computation via a single call (there is no "plan" or pre-storing of kernel matrices), the efficient use of multi-core architectures, and simplicity of the codes, installation, and interface.
In particular, in the single-core setting we are approximately 8x faster than the (single-core) CMCL library when requesting many digits in 3D.
Preliminary tests suggest that in the multi-core setting we are faster than the [Chemnitz NFFT](https://www-user.tu-chemnitz.de/~potts/nfft/) at comparable accuracy, and our code does not require an additional plan or precomputation phase.

See the manual for a more detailed description and usage.

### Dependencies

This library is only supported for unix/linux, and partially for Mac OSX.
It should be able to be compiled on Windows.

For the basic libraries

- C++ compiler such as g++ packaged with GCC
- FFTW3
- GNU make

Optional:

- numdiff (preferred but not essential; enables pass-fail math validation)
- for Fortran wrappers: compiler such as gfortran
- for matlab/octave wrappers: MATLAB, or octave and its development libs
- for building new matlab/octave wrappers (experts only): mwrap
- for the python wrappers you will need python and pip, or python3 and pip3

See [installation instructions](INSTALL.md).

### Installation and usage overview

Clone using git (or checkout using svn, or download as a zip -- see green button above), then follow the detailed [installation instructions](INSTALL.md).
If there is an error in testing, consider filing a bug report (below).

The main library is found in `lib`.
In your C++ code you will need to include the header `src/finufft.h`,
then link to the static library by compiling with `-std=c++11 -fopenmp lib/libfinufft.a -lfftw3_threads -lfftw3 -lm` for the default multi-threaded version, or
`-std=c++11 lib/libfinufft.a -lfftw3 -lm` if you edited the makefile for single-threaded.

`make examples` to compile and run the examples for calling from C++ and from C.
See [installation instructions](INSTALL.md) to build the wrappers to high-level languages (python and MATLAB/octave).


### Contents of this package

 `src` : main library source and headers. Compiled .o will be built here  
 `lib` : compiled library will be built here  
 `makefile` : GNU makefile (user may need to edit)  
 `test` : validation and performance tests. `test/check_finufft.sh` is the main validation script. `test/nuffttestnd.sh` is the main performance test script  
 `test/results` : validation comparison outputs (\*.refout; do not remove these), and local test and performance outputs (\*.out; one may remove these)  
 `examples` : simple example codes for calling the library from C++ and from C  
 `fortran` : wrappers and drivers for Fortran  
 `matlab` : wrappers and examples for MATLAB/octave  
 `matlab-mcwrap` : old mcwrap-style wrappers and examples for MATLAB  
 `finufftpy` : python wrappers  
 `python_tests` : accuracy and speed tests and examples using the python wrappers  
 `contrib` : 3rd-party code  
 `doc` : contains the manual  
 `setup.py` : enables building and installing of the python wrappers using pip or pip3  
 `README.md` : this file  
 `INSTALL.md` : installation instructions for various operating systems  
 `LICENSE` : how you may use this software  
 `CHANGELOG` : list of changes, release notes  
 `TODO` : list of things needed to do, or wishlist  

### Design notes

C++ is used for all main libraries, almost entirely avoiding object-oriented code. C++ `std::complex<double>` (aliased to `dcomplex`) and FFTW complex types are mixed within the library, since to some extent it is a glorified driver for FFTW. The interfaces are dcomplex. FFTW was considered universal and essential enough to be a dependency for the whole package.

The default FFTW plan is `FFTW_ESTIMATE`; however if you will be making multiple calls, consider using `opts` to set `fftw=FFTW_MEASURE`, which will spend many seconds planning but give the fastest speed when called again. Note that FFTW plans are saved automatically from call to call in the same executable, and the same MATLAB session.

There is a hard-defined limit of `1e11` for internal FFT arrays, set in `common.h`;
if your machine has RAM of order 1TB, and you need it, set this larger and recompile. The point of this is to catch ridiculous-sized mallocs and exit gracefully.
Note that mallocs smaller than this, but which still exceed available RAM, cause segfaults as usual. For simplicity of code, we do not do error checking on every malloc.

As a spreading kernel function, we use a new faster simplification of the Kaiser--Bessel kernel. At high requested precisions, like the Kaiser--Bessel, this achieves roughly half the kernel width achievable by a truncated Gaussian. Our kernel is exp(-beta.sqrt(1-(2x/W)^2)), where W = nspread is the full kernel width in grid units. This (and Kaiser--Bessel) are good approximations to the prolate spheroidal wavefunction of order zero (PSWF), being the functions of given support [-W/2,W/2] whose Fourier transform has minimal L2 norm outside a symmetric interval. The PSWF frequency parameter (see [ORZ]) is c = pi.(1-1/2R).W where R is the upsampling parameter (currently R=2.0).

References for this include:

[FIN] Finufft: a fast and lightweight non-uniform fast Fourier transform library. A. H. Barnett and J. F. Magland. In preparation (2017).

[ORZ] Prolate Spheroidal Wave Functions of Order Zero: Mathematical Tools for Bandlimited Approximation.  A. Osipov, V. Rokhlin, and H. Xiao. Springer (2013).

[KK] Chapter 7. System Analysis By Digital Computer. F. Kuo and J. F. Kaiser. Wiley (1967).

[FS] Nonuniform fast Fourier transforms using min-max interpolation.
J. A. Fessler and B. P. Sutton. IEEE Trans. Sig. Proc., 51(2):560-74, (Feb. 2003)

This code builds upon the CMCL NUFFT, and the Fortran wrappers duplicate its interfaces. For this the following are references:

[GL] Accelerating the Nonuniform Fast Fourier Transform. L. Greengard and J.-Y. Lee. SIAM Review 46, 443 (2004).

[LG] The type 3 nonuniform FFT and its applications. J.-Y. Lee and L. Greengard. J. Comput. Phys. 206, 1 (2005).

The original NUFFT analysis using truncated Gaussians is:

[DR] Fast Fourier Transforms for Nonequispaced data. A. Dutt and V. Rokhlin. SIAM J. Sci. Comput. 14, 1368 (1993).


### Packaged codes.

The main distribution includes contributed code by:

- Nick Hale and John Burkardt - Gauss-Legendre nodes and weights (in `contrib/`)

- Leslie Greengard and June-Yub Lee - fortran driver codes from CMCL (in `fortran/`)

### Known issues

When requestes accuracy is 1e-14 or less, it is sometimes not possible to match
this, especially when there are a large number of input and/or output points.
This is believed to be unavoidable round-off error.

Currently in Mac OSX, `make lib` fails to make the shared object library (.so).

The timing of FFTW calls is complicated, depending on
whether FFTW_ESTIMATE (the default) or FFTW_MEASURE is used.
Such issues are known, and discussed in other documentation, eg
https://pythonhosted.org/poppy/fft_optimization.html


### Bug reports

If you think you have found a bug, please contact Alex Barnett (`alex.h.barnett`
at-sign `gmail.com`) with FINUFFT in the subject line.
Include a minimal code which reproduces the bug, along with
details about your machine, operating system, compiler, and version of FINUFFT.

### Acknowledgments

The following people have helped this project through discussions, code, or bug reports:

- Leslie Greengard - CMCL test codes, testing on Mac OSX
- Dan Foreman-Mackey - python wrappers
- Charlie Epstein - discussion re analysis of kernel FT
- Andras Pataki - complex number speed in C++
- Marina Spivak - fortran testing
- Christian Muller - optimization (CMA-ES) for early kernel design
- Timo Heister - pass/fail numdiff testing ideas
- Hannah Lawrence - user testing and finding bugs
- Zydrunas Gimbutas - discussion that NFFT uses Kaiser-Bessel backwards
