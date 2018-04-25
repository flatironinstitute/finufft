Usage and interfaces
====================

In your C++ code you will need to include the header ``src/finufft.h``.
This is illustrated by the simple code ``example1d1.cpp``, in the ``examples``
directory.
From there, basic double-precision compilation with the static library is via::

  g++ example1d1.cpp -o example1d1 ../lib-static/libfinufft.a -fopenmp -lfftw3_threads -lfftw3 -lm

for the default multi-threaded version, or::
    
  g++ example1d1.cpp -o example1d1 ../lib-static/libfinufft.a -lfftw3 -lm

if you compiled FINUFFT for single-threaded only.


Interfaces from C++
*******************

You will see in  ``examples/example1d1.cpp`` the line::

  nufft_opts opts; finufft_default_opts(opts);

This is the recommended way to initialize the structure ``nufft_opts``
(defined in the header ``../src/finufft.h``).
You may override these default settings by changing the fields in this struct.
This allows control
of various parameters such as the mode ordering, FFTW plan type, and
text debug options (do not adjust ``R``; all others are fair game).
Then using the library is a matter of filling your input arrays,
allocating the correct output array size, and calling one of the
transform routines below.

We provide Type 1 (nonuniform to uniform), Type 2 (uniform to
nonuniform), and Type 3 (nonuniform to nonuniform), in dimensions 1,
2, and 3.  This gives nine routines in all.

Note on the field ``spread_sort``: the default setting is ``spread_sort=2``
which applies the following heuristic rule: in 2D or 3D always sort, but in 1D,
only sort if N (number of modes) > M/10 (where M is number of nonuniform pts).

In the below, double-precision is assumed.
See next section for single-precision compilation.

1D transforms
~~~~~~~~~~~~~

::

  int finufft1d1(BIGINT nj,double* xj,dcomplex* cj,int iflag,double eps,BIGINT ms,
	       dcomplex* fk, nufft_opts opts)

  Type-1 1D complex nonuniform FFT.

               nj-1
     fk(k1) =  SUM cj[j] exp(+/-i k1 xj(j))  for -ms/2 <= k1 <= (ms-1)/2
               j=0                            
   Inputs:
     nj     number of sources (integer of type INT; see utils.h)
     xj     nonuniform source locations, on interval [-pi,pi].
     cj     size-nj double complex array of source strengths
            (ie, stored as 2*nj doubles interleaving Re, Im).
     iflag  if >=0, uses + sign in exponential, otherwise - sign.
     eps    precision requested (>1e-16)
     ms     number of Fourier modes computed, may be even or odd;
            in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     size-ms double complex array of Fourier transform values
            stored as alternating Re & Im parts (2*ms doubles), in
 	    order determined by opts.modeord.

     The type 1 NUFFT proceeds in three main steps (see [GL]):
     1) spread data to oversampled regular mesh using kernel.
     2) compute FFT on uniform mesh
     3) deconvolve by division of each Fourier mode independently by the kernel
        Fourier series coeffs (not merely FFT of kernel), shuffle to output.

   Written with FFTW style complex arrays. Step 3a internally uses dcomplex,
   and Step 3b internally uses real arithmetic and FFTW style complex.
   Because of the former, compile with -Ofast in GNU.



  int finufft1d2(BIGINT nj,double* xj,dcomplex* cj,int iflag,double eps,BIGINT ms,
	       dcomplex* fk, nufft_opts opts)

  Type-2 1D complex nonuniform FFT.

     cj[j] = SUM   fk[k1] exp(+/-i k1 xj[j])      for j = 0,...,nj-1
             k1 
     where sum is over -ms/2 <= k1 <= (ms-1)/2.

   Inputs:
     nj     number of target (integer of type INT; see utils.h)
     xj     location of targets on interval [-pi,pi].
     fk     complex Fourier transform values (size ms)
            (ie, stored as 2*nj doubles interleaving Re, Im).
     iflag  if >=0, uses + sign in exponential, otherwise - sign.
     eps    precision requested (>1e-16)
     ms     number of Fourier modes input, may be even or odd;
            in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
     opts   struct controlling options (see finufft.h)
   Outputs:
     cj     complex double array of nj answers at targets

     The type 2 algorithm proceeds in three main steps (see [GL]).
     1) deconvolve (amplify) each Fourier mode, dividing by kernel Fourier coeff
     2) compute inverse FFT on uniform fine grid
     3) spread (dir=2, ie interpolate) data to regular mesh
     The kernel coeffs are precomputed in what is called step 0 in the code.

   Written with FFTW style complex arrays. Step 0 internally uses dcomplex,
   and Step 1 internally uses real arithmetic and FFTW style complex.
   Because of the former, compile with -Ofast in GNU.



  int finufft1d3(BIGINT nj,double* xj,dcomplex* cj,int iflag, double eps,
                 BIGINT nk, double* s, dcomplex* fk, nufft_opts opts)

  Type-3 1D complex nonuniform FFT.

               nj-1
     fk[k]  =  SUM   c[j] exp(+-i s[k] xj[j]),      for k = 0, ..., nk-1
               j=0
   Inputs:
     nj     number of sources (integer of type INT; see utils.h)
     xj     location of sources in R (real line).
     cj     size-nj double complex array of source strengths
            (ie, stored as 2*nj doubles interleaving Re, Im).
     nk     number of frequency target points
     s      frequency locations of targets in R.
     iflag  if >=0, uses + sign in exponential, otherwise - sign.
     eps    precision requested (>1e-16)
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     size-nk double complex Fourier transform values at target
            frequencies sk

     The type 3 algorithm is basically a type 2 (which is implemented precisely
     as call to type 2) replacing the middle FFT (Step 2) of a type 1. See [LG].
     Beyond this, the new twists are:
     i) nf1, number of upsampled points for the type-1, depends on the product
       of interval widths containing input and output points (X*S).
     ii) The deconvolve (post-amplify) step is division by the Fourier transform
       of the scaled kernel, evaluated on the *nonuniform* output frequency
       grid; this is done by direct approximation of the Fourier integral
       using quadrature of the kernel function times exponentials.
     iii) Shifts in x (real) and s (Fourier) are done to minimize the interval
       half-widths X and S, hence nf1.

   No references to FFTW are needed here. dcomplex arithmetic is used,
   thus compile with -Ofast in GNU.


2D transforms
~~~~~~~~~~~~~

::

  int finufft2d1(BIGINT nj,double* xj,double *yj,dcomplex* cj,int iflag,
	       double eps, BIGINT ms, BIGINT mt, dcomplex* fk, nufft_opts opts)

  Type-1 2D complex nonuniform FFT.

                   nj-1
     f[k1,k2] =    SUM  c[j] exp(+-i (k1 x[j] + k2 y[j]))
                   j=0
 
     for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2.

     The output array is k1 (fast), then k2 (slow), with each dimension
     determined by opts.modeord.
     If iflag>0 the + sign is used, otherwise the - sign is used,
     in the exponential.
                           
   Inputs:
     nj     number of sources (integer of type INT; see utils.h)
     xj,yj     x,y locations of sources on 2D domain [-pi,pi]^2.
     cj     size-nj complex double array of source strengths, 
            (ie, stored as 2*nj doubles interleaving Re, Im).
     iflag  if >=0, uses + sign in exponential, otherwise - sign.
     eps    precision requested (>1e-16)
     ms,mt  number of Fourier modes requested in x and y; each may be even or
            odd; in either case the modes are integers in [-m/2, (m-1)/2]
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     complex double array of Fourier transform values
            (size ms*mt, increasing fast in ms then slow in mt,
            ie Fortran ordering).

     The type 1 NUFFT proceeds in three main steps (see [GL]):
     1) spread data to oversampled regular mesh using kernel.
     2) compute FFT on uniform mesh
     3) deconvolve by division of each Fourier mode independently by the
        Fourier series coefficient of the kernel.
     The kernel coeffs are precomputed in what is called step 0 in the code.



  int finufft2d2(BIGINT nj,double* xj,double *yj,dcomplex* cj,int iflag,double eps,
	       BIGINT ms, BIGINT mt, dcomplex* fk, nufft_opts opts)

   Type-2 2D complex nonuniform FFT.

     cj[j] =  SUM   fk[k1,k2] exp(+/-i (k1 xj[j] + k2 yj[j]))      for j = 0,...,nj-1
             k1,k2 
     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2, 

   Inputs:
     nj     number of sources (integer of type INT; see utils.h)
     xj,yj     x,y locations of sources on 2D domain [-pi,pi]^2.
     fk     double complex array of Fourier transform values (size ms*mt,
            increasing fast in ms then slow in mt, ie Fortran ordering),
     iflag  if >=0, uses + sign in exponential, otherwise - sign.
     eps    precision requested (>1e-16)
     ms,mt  numbers of Fourier modes given in x and y; each may be even or odd;
            in either case the mode range is integers lying in [-m/2, (m-1)/2].
     opts   struct controlling options (see finufft.h)
   Outputs:
     cj     size-nj complex double array of source strengths

     The type 2 algorithm proceeds in three main steps (see [GL]).
     1) deconvolve (amplify) each Fourier mode, dividing by kernel Fourier coeff
     2) compute inverse FFT on uniform fine grid
     3) spread (dir=2, ie interpolate) data to regular mesh
     The kernel coeffs are precomputed in what is called step 0 in the code.



  int finufft2d3(BIGINT nj,double* xj,double* yj,dcomplex* cj,int iflag,
      double eps, BIGINT nk, double* s, double *t, dcomplex* fk, nufft_opts opts)

   Type-3 2D complex nonuniform FFT.

               nj-1
     fk[k]  =  SUM   c[j] exp(+-i (s[k] xj[j] + t[k] yj[j]),    for k=0,...,nk-1
               j=0
   Inputs:
     nj     number of sources (integer of type INT; see utils.h)
     xj,yj  x,y location of sources in R^2.
     cj     size-nj complex double array of source strengths, 
            (ie, stored as 2*nj doubles interleaving Re, Im).
     nk     number of frequency target points
     s,t    (k_x,k_y) frequency locations of targets in R^2.
     iflag  if >=0, uses + sign in exponential, otherwise - sign.
     eps    precision requested (>1e-16)
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     complex double Fourier transform values at the target frequencies sk

     The type 3 algorithm is basically a type 2 (which is implemented precisely
     as call to type 2) replacing the middle FFT (Step 2) of a type 1. See [LG].
     Beyond this, the new twists are:
     i) number of upsampled points for the type-1 in each dim, depends on the
       product of interval widths containing input and output points (X*S), for
       that dim.
     ii) The deconvolve (post-amplify) step is division by the Fourier transform
       of the scaled kernel, evaluated on the *nonuniform* output frequency
       grid; this is done by direct approximation of the Fourier integral
       using quadrature of the kernel function times exponentials.
     iii) Shifts in x (real) and s (Fourier) are done to minimize the interval
       half-widths X and S, hence nf, in each dim.

   No references to FFTW are needed here. Some dcomplex arithmetic is used,
   thus compile with -Ofast in GNU.


3D transforms
~~~~~~~~~~~~~

::

  int finufft3d1(BIGINT nj,double* xj,double *yj,double *zj,dcomplex* cj,int iflag,
	       double eps, BIGINT ms, BIGINT mt, BIGINT mu, dcomplex* fk,
	       nufft_opts opts)

   Type-1 3D complex nonuniform FFT.

                      nj-1
     f[k1,k2,k3] =    SUM  c[j] exp(+-i (k1 x[j] + k2 y[j] + k3 z[j]))
                      j=0

	for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2,
            -mu/2 <= k3 <= (mu-1)/2.

     In the output array, k1 is fastest, k2 middle, and k3 slowest, ie
     Fortran ordering, with each dimension
     determined by opts.modeord. If iflag>0 the + sign is used, otherwise the -
     sign is used, in the exponential.
                           
   Inputs:
     nj     number of sources (integer of type INT; see utils.h)
     xj,yj,zj   x,y,z locations of sources on 3D domain [-pi,pi]^3.
     cj     size-nj complex double array of source strengths, 
            (ie, stored as 2*nj doubles interleaving Re, Im).
     iflag  if >=0, uses + sign in exponential, otherwise - sign.
     eps    precision requested
     ms,mt,mu  number of Fourier modes requested in x,y,z;
            each may be even or odd;
            in either case the mode range is integers lying in [-m/2, (m-1)/2]
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     complex double array of Fourier transform values (size ms*mt*mu,
            increasing fast in ms to slowest in mu, ie Fortran ordering).

     The type 1 NUFFT proceeds in three main steps (see [GL]):
     1) spread data to oversampled regular mesh using kernel.
     2) compute FFT on uniform mesh
     3) deconvolve by division of each Fourier mode independently by the
        Fourier series coefficient of the kernel.
     The kernel coeffs are precomputed in what is called step 0 in the code.



  int finufft3d2(BIGINT nj,double* xj,double *yj,double *zj,dcomplex* cj,
	       int iflag,double eps, BIGINT ms, BIGINT mt, BIGINT mu,
	       dcomplex* fk, nufft_opts opts)

   Type-2 3D complex nonuniform FFT.

     cj[j] =    SUM   fk[k1,k2,k3] exp(+/-i (k1 xj[j] + k2 yj[j] + k3 zj[j]))
             k1,k2,k3
      for j = 0,...,nj-1
     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2, 
                       -mu/2 <= k3 <= (mu-1)/2

   Inputs:
     nj     number of sources (integer of type INT; see utils.h)
     xj,yj,zj     x,y,z locations of sources on 3D domain [-pi,pi]^3.
     fk     double complex array of Fourier series values (size ms*mt*mu,
            fastest in ms to slowest in mu, ie Fortran ordering).
            (ie, stored as alternating Re & Im parts, 2*ms*mt*mu doubles)
     iflag  if >=0, uses + sign in exponential, otherwise - sign.
     eps    precision requested
     ms,mt,mu  numbers of Fourier modes given in x,y,z; each may be even or odd;
            in either case the mode range is integers lying in [-m/2, (m-1)/2].
     opts   struct controlling options (see finufft.h)
   Outputs:
     cj     size-nj complex double array of target values,
            (ie, stored as 2*nj doubles interleaving Re, Im).

     The type 2 algorithm proceeds in three main steps (see [GL]).
     1) deconvolve (amplify) each Fourier mode, dividing by kernel Fourier coeff
     2) compute inverse FFT on uniform fine grid
     3) spread (dir=2, ie interpolate) data to regular mesh
     The kernel coeffs are precomputed in what is called step 0 in the code.



  int finufft3d3(BIGINT nj,double* xj,double* yj,double *zj, dcomplex* cj,
	       int iflag, double eps, BIGINT nk, double* s, double *t,
	       double *u, dcomplex* fk, nufft_opts opts)

   Type-3 3D complex nonuniform FFT.

               nj-1
     fk[k]  =  SUM   c[j] exp(+-i (s[k] xj[j] + t[k] yj[j] + u[k] zj[j]),
               j=0
                          for k=0,...,nk-1
   Inputs:
     nj     number of sources (integer of type INT; see utils.h)
     xj,yj,zj   x,y,z location of sources in R^3.
     cj     size-nj complex double array of source strengths
            (ie, interleaving Re & Im parts)
     nk     number of frequency target points
     s,t,u      (k_x,k_y,k_z) frequency locations of targets in R^3.
     iflag  if >=0, uses + sign in exponential, otherwise - sign.
     eps    precision requested
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     size-nk complex double array of Fourier transform values at the
            target frequencies sk

     The type 3 algorithm is basically a type 2 (which is implemented precisely
     as call to type 2) replacing the middle FFT (Step 2) of a type 1. See [LG].
     Beyond this, the new twists are:
     i) number of upsampled points for the type-1 in each dim, depends on the
       product of interval widths containing input and output points (X*S), for
       that dim.
     ii) The deconvolve (post-amplify) step is division by the Fourier transform
       of the scaled kernel, evaluated on the *nonuniform* output frequency
       grid; this is done by direct approximation of the Fourier integral
       using quadrature of the kernel function times exponentials.
     iii) Shifts in x (real) and s (Fourier) are done to minimize the interval
       half-widths X and S, hence nf, in each dim.

   No references to FFTW are needed here. Some dcomplex arithmetic is used,
   thus compile with -Ofast in GNU.



Custom library compilation options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may want to make the library for other data types. Currently
library names are distinct for single precision (libfinufftf) vs
double (libfinufft). However, single-threaded vs multithreaded are
built with the same name, so you will have to move them to other
locations, or build a 2nd copy of the repo, if you want to keep both
versions for use together.

You *must* do at least ``make objclean`` before changing PREC or OMP options.

a) Use ``make [task] PREC=SINGLE`` for single-precision, otherwise will be
   double-precision. Single-precision saves half the RAM, and increases
   speed slightly (<20%). The  C++, C, and fortran demos are all tested in
   single precision. However, it will break matlab, octave, python interfaces.
b) make with ``OMP=OFF`` for single-threaded, otherwise multi-threaded (openmp).
c) If you want to restrict to array sizes <2^31 and explore if 32-bit integer
   indexing beats 64-bit, add flag ``-DSMALLINT`` to ``CXXFLAGS`` which sets
   ``BIGINT`` to ``int``.
d) If you want 32 bit integers in the FINUFFT library interface instead of
   ``int64``, add flag ``-DINTERFACE32`` (experimental; C,F,M,O interfaces
   will break)

More information about large arrays:

By default FINUFFT uses 64-bit integers internally and for interfacing;
this means arguments such as the number of sources (``nj``) are type int64_t,
allowing ``nj`` to equal or exceed 2^31 (around 2e9).

There is a chance the user may want to compile a custom version with
32-bit integers internally (although we have not noticed a speed
increase on a modern CPU). In the makefile one may add the compile
flag ``-DSMALLINT`` for this, which changes ``BIGINT`` from ``int64_t`` to ``int``.

Similarly, the user may want to change the integer interface type to
32-bit ints. The compile flag ``-DINTERFACE32`` does this, and changes ``INT``
from ``int64_t`` to ``int``.

See ``../src/utils.h`` for these typedefs.

Sizes >=2^31 have been tested for C++ drivers (``test/finufft?d_test.cpp``), and
work fine, if you have enough RAM.

In fortran and C the interface is still 32-bit integers, limiting to
array sizes <2^31.

In Matlab/MEX, mwrap uses ``int`` types, so that output arrays can *only*
be <2^31.
However, input arrays >=2^31 have been tested, and while they don't crash,
they result in wrong answers (all zeros). This is yet to be fixed.

As you can see, there are some issues to clean up with large arrays and non-standard sizes. Please contribute simple solutions.


Design notes and advanced usage
*******************************

C++ is used for all main libraries, almost entirely avoiding object-oriented code. C++ ``std::complex<double>`` (aliased to ``dcomplex``) and FFTW complex types are mixed within the library, since to some extent it is a glorified driver for FFTW. The interfaces are dcomplex. FFTW was considered universal and essential enough to be a dependency for the whole package.

The default FFTW plan is ``FFTW_ESTIMATE``; however if you will be making multiple calls, consider using ``opts`` to set ``fftw=FFTW_MEASURE``, which will spend many seconds planning but give the fastest speed when called again. Note that FFTW plans are saved automatically from call to call in the same executable, and the same MATLAB session.

There is a hard-defined limit of ``1e11`` for internal FFT arrays, set in ``common.h``;
if your machine has RAM of order 1TB, and you need it, set this larger and recompile. The point of this is to catch ridiculous-sized mallocs and exit gracefully.
Note that mallocs smaller than this, but which still exceed available RAM, cause segfaults as usual. For simplicity of code, we do not do error checking on every malloc.

As a spreading kernel function, we use a new faster simplification of the Kaiser--Bessel kernel. At high requested precisions, like the Kaiser--Bessel, this achieves roughly half the kernel width achievable by a truncated Gaussian. Our kernel is exp(-beta.sqrt(1-(2x/W)^2)), where W = nspread is the full kernel width in grid units. This (and Kaiser--Bessel) are good approximations to the prolate spheroidal wavefunction of order zero (PSWF), being the functions of given support [-W/2,W/2] whose Fourier transform has minimal L2 norm outside a symmetric interval. The PSWF frequency parameter (see [ORZ]) is c = pi.(1-1/2R).W where R is the upsampling parameter (currently R=2.0).

