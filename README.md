# Flatiron Institute Nonuniform Fast Fourier Transform libraries: FINUFFT

Version 0.7  (3/15/2017)

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

This library is only supported for unix/linux and Mac OSX right now.

For the basic libraries

- C++ compiler such as g++
- FFTW3
- GNU make
- numdiff (preferred but not essential; enables pass-fail math validation)

For the Fortran wrappers

- Fortran compiler such as gfortran (see settings in the makefile)

On an Ubuntu linux, these dependencies can be installed as follows:
```
sudo apt-get install make build-essential libfftw3-dev gfortran numdiff
```
On a Fedora/CentOS linux system:
```
sudo yum install make gcc gcc-c++ gcc-gfortran fftw3 fftw3-devel libgomp
```
then follow instructions to install [numdiff](http://www.nongnu.org/numdiff).


### Installation and usage

Clone using git (or checkout using svn, or download as a zip -- see green button above), then follow the detailed [installation instructions](INSTALL.md).

Compile and check via `make test`, which should report zero crashes and zero fails. (If numdiff was not installed, it instead produces output that you will have to check by eye matches the requested accuracy.)
If there is an error in compilation, then `cp makefile makefile.local`, edit `makefile.local` to adjust compiler and other library options, then use `make -f makefile.local`.
If there is an error in testing, consider filing a bug report (below).

The main library is found in `lib`.
In your C++ code you will need to include the header `src/finufft.h`,
then link to the static library by compiling with `-std=c++11 -fopenmp lib/libfinufft.a -lfftw3_omp -lfftw3 -lm` for the default multi-threaded version, or
`-std=c++11 lib/libfinufft.a -lfftw3 -lm` if you edited the makefile for single-threaded.

`make examples` to compile and run the examples for calling from C++ and from C.

`make fortran` to compile and run the fortran wrappers and examples.

`make matlab` to build the MEX interface to matlab



### Contents of this package

 `src` : main library source and headers. Compiled .o will be built here.  
 `lib` : compiled library will be built here.  
 `makefile` : GNU makefile (user may need to edit)  
 `test` : validation and performance tests. `test/check_finufft.sh` is the main validation script. `test/nuffttestnd.sh` is the main performance test script.  
 `test/results` : validation comparison outputs (\*.refout; do not remove these), and local test and performance outputs (\*.out; one may remove these)
 `examples` : simple example codes for calling the library from C++ and from C.  
 `fortran` : wrappers and drivers for Fortran.   
 `matlab` : wrappers and examples for MATLAB/octave. (Not yet working)  
 `matlab-mcwrap` : wrappers and examples for MATLAB.  
 `python` : wrappers and examples for python.  
 `contrib` : 3rd-party code.  
 `devel` : various obsolete or in-development codes (experts only)  
 `doc` : the manual (not yet there)  
 `README.md` : this file  
 `INSTALL.md` : installation instructions for various operating systems  
 `LICENSE` : licensing information  
 `CHANGELOG` : list of changes made  
 `TODO` : list of things needed to do, or wishlist  

### Notes

C++ is used for all main libraries, avoiding object-oriented code. C++ `std::complex<double>` (aliased to `dcomplex`) and FFTW complex types are mixed within the library, since to some extent it is a glorified driver for FFTW. The interfaces are dcomplex. FFTW was considered universal and essential enough to be a dependency for the whole package.

If internal arrays to be dynamically allocated are larger than `opts.maxnalloc`, the library stops before allocating and returns an error code. Currently the default value is `opts.maxnalloc=1e9`, which would allocate at least 16 GB. If your machine has the RAM and you need it, set this larger before calling.
(Currently changing this at runtime is only available via the C++ interface.)

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
Dan Foreman-Mackey - python wrappers (in `python/')  

There are also undocumented packaged codes in the `devel/` directory.


### Known issues

When requestes accuracy is 1e-14 or less, it is sometimes not possible to match
this, especially when there are a large number of input and/or output points.
This is believed to be unavoidable round-off error.

### Bug reports

If you think you have found a bug, please contact Alex Barnett (`ahb`
at-sign `math.dartmouth.edu`) with FINUFFT in the subject line.
Include a minimal code which reproduces the bug, along with
details about your machine, operating system, and version of FINUFFT.

### Acknowledgments

The following people have greatly helped this project either via discussions or bug reports:

Leslie Greengard  
Charlie Epstein  
Andras Pataki  
Marina Spivak  
Christina Muller  
Timo Heister  
Dan Foreman-Mackey  
