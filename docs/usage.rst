Usage and interfaces
====================

In your C++ code you will need to include the header ``src/finufft.h``.
This is illustrated by the simple code ``example1d1.cpp``, in the ``examples``
directory.
From there, basic double-precision compilation with the static library is via::

  g++ example1d1.cpp -o example1d1 ../lib-static/libfinufft.a -fopenmp -lfftw3_threads -lfftw3 -lm

for the default multi-threaded version, or, if you compiled FINUFFT for single-threaded::

  g++ example1d1.cpp -o example1d1 ../lib-static/libfinufft.a -lfftw3 -lm

The ``examples`` and ``test`` directories are good places to see usage examples.

If you have an application with multiple strength or coefficient vectors with fixed nonuniform points, see the :ref:`advanced interfaces <manyinterface>`.


Interfaces from C++
*******************

We provide Type 1 (nonuniform to uniform), Type 2 (uniform to
nonuniform), and Type 3 (nonuniform to nonuniform), in dimensions 1,
2, and 3.  This gives nine routines in all.

Using the library is a matter of filling your input arrays,
allocating the correct output array size, possibly setting fields in
the options struct, then calling one of the transform routines below.

Now, more about the options.
You will see in  ``examples/example1d1.cpp`` the line::

  nufft_opts opts; finufft_default_opts(opts);

This is the recommended way to initialize the structure ``nufft_opts``.
You may override these default settings by changing the fields in this struct.
This allows control of various parameters such as the mode ordering, FFTW plan mode,
upsampling factor :math:`\sigma`, and debug/timing output.
Here is the list of the options fields you may set (see the header ``../src/finufft.h``).
Here the abbreviation ``FLT`` means ``double`` if compiled in
the default double-precision, or ``single`` if single precision:

::

  int debug;          // 0: silent, 1: text basic timing output
  int spread_debug;   // passed to spread_opts, 0 (no text) 1 (some) or 2 (lots)
  int spread_sort;    // passed to spread_opts, 0 (don't sort) 1 (do) or 2 (heuristic)
  int spread_kerevalmeth; // "     spread_opts, 0: exp(sqrt()), 1: Horner ppval (faster)
  int spread_kerpad;  // passed to spread_opts, 0: don't pad to mult of 4, 1: do
  int chkbnds;        // 0: don't check if input NU pts in [-3pi,3pi], 1: do
  int fftw;           // 0:FFTW_ESTIMATE, or 1:FFTW_MEASURE (slow plan but faster)
  int modeord;        // 0: CMCL-style increasing mode ordering (neg to pos), or
                      // 1: FFT-style mode ordering (affects type-1,2 only)
  FLT upsampfac;      // upsampling ratio sigma, either 2.0 (standard) or 1.25 (small FFT)

Here are their default settings (set in ``../src/common.cpp:finufft_default_opts``):

::

  debug = 0;
  spread_debug = 0;
  spread_sort = 2;
  spread_kerevalmeth = 1;
  spread_kerpad = 1;
  chkbnds = 0;
  fftw = FFTW_ESTIMATE;
  modeord = 0;
  upsampfac = (FLT)2.0;

Notes on various options:

``spread_sort``: the default setting is ``spread_sort=2``
which applies the following heuristic rule: in 2D or 3D always sort, but in 1D,
only sort if N (number of modes) > M/10 (where M is number of nonuniform pts).

``fftw``:
The default FFTW plan is ``FFTW_ESTIMATE``; however if you will be making multiple calls, consider ``fftw=FFTW_MEASURE``, which will spend many seconds planning but give the fastest speed when called again. Note that FFTW plans are saved (by FFTW's library)
automatically from call to call in the same executable (incidentally also in the same MATLAB/octave or python session).

``upsampfac``: This is the internal factor by which the FFT is larger than
the number of requested modes in each dimension. We have built efficient kernels
for only two settings: ``upsampfac=2.0`` (standard), and ``upsampfac=1.25``
(lower RAM, smaller FFTs).
The latter can be much faster when the number of nonuniform points is similar or
smaller to the number of modes, and/or if low accuracy is required.
It is especially much faster for type 3 transforms.
However, the kernel widths :math:`w` are about 50% larger in each dimension,
which can lead to slower spreading (it can also be faster due to the smaller
size of the fine grid).
Thus only 9-digit accuracy can be reached with ``upsampfac=1.25``.

.. _errcodes:

Error codes
~~~~~~~~~~~

In the interfaces, the returned value is 0 if successful, otherwise the error code
has the following meanings (see ``../src/utils.h``):

::

  1  requested tolerance epsilon too small
  2  attemped to allocate internal arrays larger than MAX_NF (defined in common.h)
  3  spreader: fine grid too small
  4  spreader: if chkbnds=1, a nonuniform point out of input range [-3pi,3pi]^d
  5  spreader: array allocation error
  6  spreader: illegal direction (should be 1 or 2)
  7  upsampfac too small (should be >1)
  8  upsampfac not a value with known Horner eval: currently 2.0 or 1.25 only
  9  ndata not valid (should be >= 1)

In the interfaces below, ``int64`` (typedefed as ``BIGINT`` in the code)
means 64-bit signed integer type, ie ``int64_t``.
This is used for all potentially large integers, in case the user wants
large problems involving more than 2^31 points.
``int`` is the usual 32-bit signed integer.
The ``FLT`` type is, as above, either ``double`` or ``single``.


1D transforms
~~~~~~~~~~~~~

::

  int finufft1d1(int64 nj,double* xj,dcomplex* cj,int iflag,double eps,int64 ms,
	       dcomplex* fk, nufft_opts opts)

  Type-1 1D complex nonuniform FFT.

               nj-1
     fk(k1) =  SUM cj[j] exp(+/-i k1 xj(j))  for -ms/2 <= k1 <= (ms-1)/2
               j=0
  Inputs:
     nj     number of sources (int64)
     xj     location of sources (size-nj FLT array), in [-3pi,3pi]
     cj     size-nj FLT complex array of source strengths
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
     eps    precision requested (>1e-16)
     ms     number of Fourier modes computed, may be even or odd (int64);
            in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     size-ms FLT complex array of Fourier transform values
            stored as alternating Re & Im parts (2*ms FLTs)
 	    order determined by opts.modeord.
     returned value - 0 if success, else see ../docs/usage.rst

     The type 1 NUFFT proceeds in three main steps (see [GL]):
     1) spread data to oversampled regular mesh using kernel.
     2) compute FFT on uniform mesh
     3) deconvolve by division of each Fourier mode independently by the kernel
        Fourier series coeffs (not merely FFT of kernel), shuffle to output.

   Written with FFTW style complex arrays. Step 3a internally uses dcomplex,
   and Step 3b internally uses real arithmetic and FFTW style complex.
   Because of the former, compile with -Ofast in GNU.



  int finufft1d2(int64 nj,double* xj,dcomplex* cj,int iflag,double eps,int64 ms,
	       dcomplex* fk, nufft_opts opts)

  Type-2 1D complex nonuniform FFT.

     cj[j] = SUM   fk[k1] exp(+/-i k1 xj[j])      for j = 0,...,nj-1
             k1
     where sum is over -ms/2 <= k1 <= (ms-1)/2.

   Inputs:
     nj     number of targets (int64)
     xj     location of targets (size-nj FLT array), in [-3pi,3pi]
     fk     complex Fourier transform values (size ms, ordering set by opts.modeord)
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     iflag  if >=0, uses + sign in exponential, otherwise - sign (int).
     eps    precision requested (>1e-16)
     ms     number of Fourier modes input, may be even or odd (int64);
            in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
     opts   struct controlling options (see finufft.h)
   Outputs:
     cj     complex FLT array of nj answers at targets
     returned value - 0 if success, else see ../docs/usage.rst

     The type 2 algorithm proceeds in three main steps (see [GL]).
     1) deconvolve (amplify) each Fourier mode, dividing by kernel Fourier coeff
     2) compute inverse FFT on uniform fine grid
     3) spread (dir=2, ie interpolate) data to regular mesh
     The kernel coeffs are precomputed in what is called step 0 in the code.

   Written with FFTW style complex arrays. Step 0 internally uses dcomplex,
   and Step 1 internally uses real arithmetic and FFTW style complex.
   Because of the former, compile with -Ofast in GNU.



  int finufft1d3(int64 nj,double* xj,dcomplex* cj,int iflag, double eps,
                 int64 nk, double* s, dcomplex* fk, nufft_opts opts)

  Type-3 1D complex nonuniform FFT.

               nj-1
     fk[k]  =  SUM   c[j] exp(+-i s[k] xj[j]),      for k = 0, ..., nk-1
               j=0
   Inputs:
     nj     number of sources (int64)
     xj     location of sources on real line (nj-size array of FLT)
     cj     size-nj FLT complex array of source strengths
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     nk     number of frequency target points (int64)
     s      frequency locations of targets in R.
     iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
     eps    precision requested (>1e-16)
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     size-nk FLT complex Fourier transform values at target
            frequencies sk
     returned value - 0 if success, else see ../docs/usage.rst

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


2D transforms
~~~~~~~~~~~~~

::

  int finufft2d1(int64 nj,double* xj,double *yj,dcomplex* cj,int iflag,
	       double eps, int64 ms, int64 mt, dcomplex* fk, nufft_opts opts)

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
     nj     number of sources (int64)
     xj,yj     x,y locations of sources (each a size-nj FLT array) in [-3pi,3pi]
     cj     size-nj complex FLT array of source strengths,
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
     eps    precision requested (>1e-16)
     ms,mt  number of Fourier modes requested in x and y (int64);
            each may be even or odd;
            in either case the mode range is integers lying in [-m/2, (m-1)/2]
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     complex FLT array of Fourier transform values
            (size ms*mt, fast in ms then slow in mt,
            ie Fortran ordering).
     returned value - 0 if success, else see ../docs/usage.rst

     The type 1 NUFFT proceeds in three main steps (see [GL]):
     1) spread data to oversampled regular mesh using kernel.
     2) compute FFT on uniform mesh
     3) deconvolve by division of each Fourier mode independently by the
        Fourier series coefficient of the kernel.
     The kernel coeffs are precomputed in what is called step 0 in the code.



  int finufft2d2(int64 nj,double* xj,double *yj,dcomplex* cj,int iflag,double eps,
	       int64 ms, int64 mt, dcomplex* fk, nufft_opts opts)

   Type-2 2D complex nonuniform FFT.

     cj[j] =  SUM   fk[k1,k2] exp(+/-i (k1 xj[j] + k2 yj[j]))      for j = 0,...,nj-1
             k1,k2
     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2,

    Inputs:
     nj     number of targets (int64)
     xj,yj     x,y locations of targets (each a size-nj FLT array) in [-3pi,3pi]
     fk     FLT complex array of Fourier transform values (size ms*mt,
            changing fast in ms then slow in mt, as in Fortran)
            Along each dimension the ordering is set by opts.modeord.
     iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
     eps    precision requested (>1e-16)
     ms,mt  numbers of Fourier modes given in x and y (int64)
            each may be even or odd;
            in either case the mode range is integers lying in [-m/2, (m-1)/2].
     opts   struct controlling options (see finufft.h)
   Outputs:
     cj     size-nj complex FLT array of target values
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     returned value - 0 if success, else see ../docs/usage.rst

     The type 2 algorithm proceeds in three main steps (see [GL]).
     1) deconvolve (amplify) each Fourier mode, dividing by kernel Fourier coeff
     2) compute inverse FFT on uniform fine grid
     3) spread (dir=2, ie interpolate) data to regular mesh
     The kernel coeffs are precomputed in what is called step 0 in the code.



  int finufft2d3(int64 nj,double* xj,double* yj,dcomplex* cj,int iflag,
      double eps, int64 nk, double* s, double *t, dcomplex* fk, nufft_opts opts)

   Type-3 2D complex nonuniform FFT.

               nj-1
     fk[k]  =  SUM   c[j] exp(+-i (s[k] xj[j] + t[k] yj[j]),    for k=0,...,nk-1
               j=0
   Inputs:
     nj     number of sources (int64)
     xj,yj  x,y location of sources in the plane R^2 (each size-nj FLT array)
     cj     size-nj complex FLT array of source strengths,
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     nk     number of frequency target points (int64)
     s,t    (k_x,k_y) frequency locations of targets in R^2.
     iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
     eps    precision requested (>1e-16)
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     size-nk complex FLT Fourier transform values at the
            target frequencies sk
     returned value - 0 if success, else see ../docs/usage.rst

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


3D transforms
~~~~~~~~~~~~~

::

  int finufft3d1(int64 nj,double* xj,double *yj,double *zj,dcomplex* cj,int iflag,
	       double eps, int64 ms, int64 mt, int64 mu, dcomplex* fk,
	       nufft_opts opts)

   Type-1 3D complex nonuniform FFT.

                      nj-1
     f[k1,k2,k3] =    SUM  c[j] exp(+-i (k1 x[j] + k2 y[j] + k3 z[j]))
                      j=0

	for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2,
            -mu/2 <= k3 <= (mu-1)/2.

      The output array is as in opt.modeord in each dimension.
     k1 changes is fastest, k2 middle,
     and k3 slowest, ie Fortran ordering. If iflag>0 the + sign is
     used, otherwise the - sign is used, in the exponential.

   Inputs:
     nj     number of sources (int64)
     xj,yj,zj   x,y,z locations of sources (each size-nj FLT array) in [-3pi,3pi]
     cj     size-nj complex FLT array of source strengths,
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
     eps    precision requested
     ms,mt,mu  number of Fourier modes requested in x,y,z (int64);
            each may be even or odd;
            in either case the mode range is integers lying in [-m/2, (m-1)/2]
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     complex FLT array of Fourier transform values (size ms*mt*mu,
            changing fast in ms to slowest in mu, ie Fortran ordering).
     returned value - 0 if success, else see ../docs/usage.rst

     The type 1 NUFFT proceeds in three main steps (see [GL]):
     1) spread data to oversampled regular mesh using kernel.
     2) compute FFT on uniform mesh
     3) deconvolve by division of each Fourier mode independently by the
        Fourier series coefficient of the kernel.
     The kernel coeffs are precomputed in what is called step 0 in the code.



  int finufft3d2(int64 nj,double* xj,double *yj,double *zj,dcomplex* cj,
	       int iflag,double eps, int64 ms, int64 mt, int64 mu,
	       dcomplex* fk, nufft_opts opts)

   Type-2 3D complex nonuniform FFT.

     cj[j] =    SUM   fk[k1,k2,k3] exp(+/-i (k1 xj[j] + k2 yj[j] + k3 zj[j]))
             k1,k2,k3
      for j = 0,...,nj-1
     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2,
                       -mu/2 <= k3 <= (mu-1)/2

   Inputs:
     nj     number of targets (int64)
     xj,yj,zj  x,y,z locations of targets (each size-nj FLT array) in [-3pi,3pi]
     fk     FLT complex array of Fourier series values (size ms*mt*mu,
            changing fastest in ms to slowest in mu, ie Fortran ordering).
	    (ie, stored as alternating Re & Im parts, 2*ms*mt*mu FLTs)
	    Along each dimension, opts.modeord sets the ordering.
     iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
     eps    precision requested
     ms,mt,mu  numbers of Fourier modes given in x,y,z (int64);
            each may be even or odd;
            in either case the mode range is integers lying in [-m/2, (m-1)/2].
     opts   struct controlling options (see finufft.h)
   Outputs:
     cj     size-nj complex FLT array of target values,
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     returned value - 0 if success, else see ../docs/usage.rst

     The type 2 algorithm proceeds in three main steps (see [GL]).
     1) deconvolve (amplify) each Fourier mode, dividing by kernel Fourier coeff
     2) compute inverse FFT on uniform fine grid
     3) spread (dir=2, ie interpolate) data to regular mesh
     The kernel coeffs are precomputed in what is called step 0 in the code.



  int finufft3d3(int64 nj,double* xj,double* yj,double *zj, dcomplex* cj,
	       int iflag, double eps, int64 nk, double* s, double *t,
	       double *u, dcomplex* fk, nufft_opts opts)

   Type-3 3D complex nonuniform FFT.

               nj-1
     fk[k]  =  SUM   c[j] exp(+-i (s[k] xj[j] + t[k] yj[j] + u[k] zj[j]),
               j=0

  Inputs:
     nj     number of sources (int64)
     xj,yj,zj     x,y,z location of sources in R^3 (each size-nj FLT array)
     cj     size-nj complex FLT array of source strengths
            (ie, interleaving Re & Im parts)
     nk     number of frequency target points (int64)
     s,t,u      (k_x,k_y,k_z) frequency locations of targets in R^3.
     iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
     eps    precision requested (FLT)
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     size-nk complex FLT array of Fourier transform values at the
            target frequencies sk
     returned value - 0 if success, else see ../docs/usage.rst
                          for k=0,...,nk-1

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



Interfaces from C
*****************

The C user should initialize the options struct via::

  nufft_c_opts opts; finufft_default_c_opts(opts);

Options fields may then be changed in ``opts`` before passing to the following interfaces. We use the C99 complex type ``_Complex``, which is the same as
``complex``. As above, ``FLT`` indicates ``double`` or ``float``.
The meaning of arguments are identical to the C++ documentation above.
For a demo see ``examples/example1d1c.c``::

  int finufft1d1_c(int nj,FLT* xj,FLT _Complex* cj,int iflag, FLT eps,int ms, FLT _Complex* fk, nufft_c_opts copts);
  int finufft1d2_c(int nj,FLT* xj,FLT _Complex* cj,int iflag, FLT eps,int ms, FLT _Complex* fk, nufft_c_opts copts);
  int finufft1d3_c(int j,FLT* x,FLT _Complex* c,int iflag,FLT eps,int nk, FLT* s, FLT _Complex* f, nufft_c_opts copts);
  int finufft2d1_c(int nj,FLT* xj,FLT *yj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt,FLT _Complex* fk, nufft_c_opts copts);
  int finufft2d1many_c(int ndata,int nj,FLT* xj,FLT *yj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt,FLT _Complex* fk, nufft_c_opts copts);
  int finufft2d2_c(int nj,FLT* xj,FLT *yj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt, FLT _Complex* fk, nufft_c_opts copts);
  int finufft2d2many_c(int ndata,int nj,FLT* xj,FLT *yj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt, FLT _Complex* fk, nufft_c_opts copts);
  int finufft2d3_c(int nj,FLT* x,FLT *y,FLT _Complex* c,int iflag,FLT eps,int nk, FLT* s, FLT *t,FLT _Complex* f, nufft_c_opts copts);
  int finufft3d1_c(int nj,FLT* xj,FLT* yj,FLT *zj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt, int mu,FLT _Complex* fk, nufft_c_opts copts);
  int finufft3d2_c(int nj,FLT* xj,FLT *yj,FLT *zj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt, int mu, FLT _Complex* fk, nufft_c_opts copts);
  int finufft3d3_c(int nj,FLT* x,FLT *y,FLT *z,FLT _Complex* c,int iflag,FLT eps,int nk, FLT* s, FLT *t,FLT *u,FLT _Complex* f, nufft_c_opts copts);


Interfaces from fortran
***********************

We have not yet included control of the options in the fortran wrappers. Please help create these if you can.
The meaning of arguments is as in the C++ documentation above,
apart from that now ``ier`` is an argument which is output to.
Examples of calling all 9 routines from fortran are in ``fortran/nufft?d_demo.f`` (for double-precision) and ``fortran/nufft?d_demof.f`` (single-precision).
Here are the calling commands with fortran types for the default double-precision case::

      integer ier,iflag,ms,mt,mu,nj,ndata
      real*8, allocatable :: xj(:),yj(:),zj(:), sk(:),tk(:),uk(:)
      real*8 err,eps
      complex*16, allocatable :: cj(:), fk(:)

      call finufft1d1_f(nj,xj,cj,iflag,eps, ms,fk,ier)
      call finufft1d2_f(nj,xj,cj,iflag, eps, ms,fk,ier)
      call finufft1d3_f(nj,xj,cj,iflag,eps, ms,sk,fk,ier)
      call finufft2d1_f(nj,xj,yj,cj,iflag,eps,ms,mt,fk,ier)
      call finufft2d1many_f(ndata,nj,xj,yj,cj,iflag,eps,ms,mt,fk,ier)
      call finufft2d2_f(nj,xj,yj,cj,iflag,eps,ms,mt,fk,ier)
      call finufft2d2many_f(ndata,nj,xj,yj,cj,iflag,eps,ms,mt,fk,ier)
      call finufft2d3_f(nj,xj,yj,cj,iflag,eps,nk,sk,tk,fk,ier)
      call finufft3d1_f(nj,xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk,ier)
      call finufft3d2_f(nj,xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk,ier)
      call finufft3d3_f(nj,xj,yj,zj,cj,iflag,eps,nk,sk,tk,uk,fk,ier)





Design notes and data types
***************************

We strongly recommend you use ``upsampfac=1.25`` for type-3; it
reduces its run-time from around 8 times the types 1 or 2, to around 3-4
times. It is often also faster for type-1 and type-2, at low precisions.

When you include the header ``finufft.h`` you have access to the ``BIGINT`` type
which is used for all potentially-large input integers (M, N, etc), and
currently typedefed to ``int64_t`` (see ``utils.h``).
This allows the number of sources, number of modes, etc,
to safely exceed 2^31 (around 2e9).
In case you were to want to change this
type, you may want to use ``BIGINT`` in your calling codes.
Using ``int64_t`` will be fine if you don't change this.
To change (perhaps for speed, but we have not noticed any speed hit using
64-bit integers throughout), one would change
``BIGINT`` from ``int64_t`` to ``int`` in ``utils.h``.

Sizes >=2^31 have been tested for C++ drivers (``test/finufft?d_test.cpp``), and
work fine, if you have enough RAM.

In fortran and C the interface is still 32-bit integers, limiting to
array sizes <2^31.

C++ is used for all main libraries, almost entirely avoiding object-oriented code. C++ ``std::complex<double>`` (aliased to ``dcomplex``) and FFTW complex types are mixed within the library, since to some extent it is a glorified driver for FFTW. The interfaces are dcomplex. FFTW was considered universal and essential enough to be a dependency for the whole package.

There is a hard-defined limit of ``1e11`` for internal FFT arrays, set in ``common.h`` as ``MAX_NF``:
if your machine has RAM of order 1TB, and you need it, set this larger and recompile. The point of this is to catch ridiculous-sized mallocs and exit gracefully.
Note that mallocs smaller than this, but which still exceed available RAM, cause segfaults as usual. For simplicity of code, we do not do error checking on every malloc.

As a spreading kernel function, we use a new faster simplification of the Kaiser--Bessel kernel. At high requested precisions, like the Kaiser--Bessel, this achieves roughly half the kernel width achievable by a truncated Gaussian. Our kernel is exp(-beta.sqrt(1-(2x/W)^2)), where W = nspread is the full kernel width in grid units. This (and Kaiser--Bessel) are good approximations to the prolate spheroidal wavefunction of order zero (PSWF), being the functions of given support [-W/2,W/2] whose Fourier transform has minimal L2 norm outside of a symmetric interval. The PSWF frequency parameter (see [ORZ]) is c = pi.(1-1/2sigma).W where sigma is the upsampling parameter. See our forthcoming paper.
