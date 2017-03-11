# Flatiron Institute Nonuniform Fast Fourier Transform libraries: FINUFFT

Version 0.7  (3/10/2017)

### Alex H. Barnett and Jeremy F. Magland

### Purpose

This is a lightweight library to compute the nonuniform FFT to a specified precision, in one, two, or three dimensions.
This task is to approximate various exponential sums involving large numbers of terms and output indices, in close to linear time.
The speedup over naive evaluation of the sums is similar to that achieved by the FFT. For instance, for _N_ terms and _N_ output indices, the computation time is _O_(_N_ log _N_) as opposed to the naive _O_(_N_<sup>2</sup>).
For convenience, we conform to the simple existing interfaces of the
[CMCL NUFFT libraries of Greengard--Lee from 2004](http://www.cims.nyu.edu/cmcl/nufft/nufft.html).
Our main innovations are: speed (enhanced by a new functional form for the spreading kernel), computation via a single call (there is no "plan" or pre-storing of kernel matrices), the efficient use of multi-core architectures, and simplicity of the codes, installation, and interface.
In particular, in the single-core setting we are approximately 8x faster than the (single-core) CMCL library when requesting many digits in 3D.
Preliminary tests suggest that in the multi-core setting we are no slower than the [Chemnitz NFFT](https://www-user.tu-chemnitz.de/~potts/nfft/) at comparable accuracy, except that our code does not require an additional plan or precomputation phase.

See the manual for more information.

### Dependencies

For the basic libraries

- C\++ compiler such as g\++
- FFTW3
- GNU make
- numdiff
- Optionally, OpenMP (however, the makefile can be adjusted for single-threaded operation)

For the Fortran wrappers

- Fortran compiler such as gfortran (see settings in the makefile)

On a Fedora/CentOS linux system, these dependencies can be installed as follows:
```bash
sudo yum install make gcc gcc-c++ gcc-gfortran fftw3 fftw3-devel libgomp
```
On Ubuntu linux:
```bash
sudo apt-get install make build-essential libfftw3-dev gfortran
```

### Installation

- Clone using git (or checkout using svn, or download as a zip -- see green button above)
- Compile the library using:

```bash
make
```
This should compile the main libraries and print a list of further make options.
If you have an error then `cp makefile makefile.local`, edit `makefile.local` to adjust
compiler and other library options, and use `make -f makefile.local`.
Here are some other make tasks:

- `make examples` : compile some simple examples in `examples/`
- `make test` : mathematical validation of the library and components
- `make perftest` : multi-threaded and single-threaded performance tests
- `make fortran` : compile and test the fortran interfaces  

Linking to the library:
In C\++ the simplest is to link to the static library by compiling with `-std=c++11 -fopenmp lib/libfinufft.a -lfftw3_omp -lfftw3 -lm` for the default multi-threaded version, or
`-std=c++11 lib/libfinufft.a -lfftw3 -lm` if you edited the makefile for single-threading.
In your C\++ code you will need to include the header `src/finufft.h`.
See codes in `examples/` for calling from C, and `fortran/` for calling from Fortran.


### Contents of this package

- `src` : main library source and headers. Compiled .o will be built here.
- `lib` : compiled library will be built here.  
- `test` : validation and performance tests.  

- `examples` : test codes (drivers) which verify libaries are working correctly, perform speed tests, and show how to call them. ***
- `examples/nuffttestnd.sh` : benchmark and display accuracy for all types and dimensions (3x3 = 9 in total) of NUFFT at fixed requested tolerance  
- `examples/checkallaccs.sh [dim]` : sweep over all tolerances checking the spreader and NUFFT at a single dimension;  [dim] is 1, 2, or 3
- `examples/results` : accuracy and timing outputs.
***
  
- `contrib` : 3rd-party code.  
- `fortran` : wrappers and drivers for Fortran.
- `matlab` : wrappers and examples for MATLAB. (Not yet working)  
- `devel` : various obsolete or in-development codes (experts only)  
- `makefile` : GNU makefile (user may need to edit)  
- `doc` : the manual (not yet there)  
- `README.md` : this file  
- `LICENSE` : licensing information  
- `CHANGELOG` : list of changes made  
- `TODO` : list of things needed to do  


### Notes

C\++ is used for all main libraries, although without much object-oriented code. C\++ complex<double> ("dcomplex") and FFTW complex types are mixed within the library, since it is a glorified driver for FFTW, but has dcomplex interfaces and test codes. FFTW was considered universal and essential enough to be a dependency for the whole package.

As a spreading kernel function, we use a new faster simplification of the Kaiser--Bessel kernel. At high requested precisions, like the Kaiser--Bessel, this achieves roughly half the kernel width achievable by a truncated Gaussian. Our kernel is exp(-beta.sqrt(1-(2x/W)^2)), where W = nspread is the full kernel width in grid units. This (and Kaiser--Bessel) are good approximations to the prolate spheroidal wavefunction of order zero (PSWF), being the functions of given support [-W/2,W/2] whose Fourier transform has minimal L2 norm outside a symmetric interval. The PSWF frequency parameter (see [ORZ]) is c = pi.(1-1/2R).W where R is the upsampling parameter (currently R=2.0).

References for this include:

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

The main distribution includes code by:

Nick Hale and John Burkardt - Gauss-Legendre nodes and weights (in `contrib/`)  
Leslie Greengard and June-Yub Lee - fortran driver codes from CMCL (in `fortran/`)  

There are also packaged codes in the `devel/` directory.


### Known issues

Currently, if a large product of x width and k width is requested in type 3 transforms,
no check is done before attempting to malloc ridiculous array sizes.


### Bug reports

If you think you have found a bug, please contact Alex Barnett (`ahb`
at-sign `math.dartmouth.edu`) with FINUFFT in the subject line.
Include code with reproduces the bug, along with
details about your machine, operating system, and version of FINUFFT.

### Acknowledgments

The following people have greatly helped this project either via discussions or bug reports:

Leslie Greengard  
Charlie Epstein  
Andras Pataki  
Marina Spivak  
Timo Heister  
