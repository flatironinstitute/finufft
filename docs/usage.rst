.. _c:

Usage from C/C++
================

Quick start guide
-----------------

We first give a simple example of performing a 1D type-1 transform
in double precision from C++, the library's native language,
using plain arrays (for simplicity)
of C++ complex number type. First include the headers::

  #include "finufft.h"
  #include <complex>

Now in the body of the code, assuming ``M`` has been set to be
the number of nonuniform points, we allocate the input arrays::

  double *x = (double *)malloc(sizeof(double)*M);
  complex<double>* c = (complex<double>*)malloc(sizeof(complex<double>)*M);
  
The user should now fill these with their input data;
values in ``x`` should lie in :math:`[-3\pi,3\pi]`, and
``c`` can be arbitrary complex strengths (we omit example code for this here).
With ``N`` as the desired number of modes, allocate an output array::

  complex<double>* F = (complex<double>*)malloc(sizeof(complex<double>)*N);

To perform the nonuniform FFT (with default options) is then one line::

  int ier = finufft1d1(M,x,c,+1,1e-6,N,F,NULL);

Here ``NULL`` is interpreted as using default options.
This fills ``F`` with the output modes, in increasing ordering
from ``-N/2`` to ``N/2-1``.
Here ``+1`` sets the sign of ``i`` in the exponentials in the
:ref:`definitions <math>`,
``1e-6`` chooses 6-digit relative tolerance, and ``ier`` is a status output
which is zero if successful (see below).

To set non-default options, first
put default values in a ``nufft_opts`` struct with pointer ``popts``,
make your changes, then pass the pointer to FINUFFT::

  nufft_opts* popts;
  finufft_default_opts(popts);
  popts->debug = 1;                                // example option choice
  int ier = finufft1d1(M,x,c,+1,1e-6,N,F,popts);
  
.. warning::
   - Without the ``finufft_default_opts`` call, options may take on random values which may cause a crash.
   - This usage has changed from versions 1.0 and 1.1 which used C++-style pass by reference. Please make sure you pass a *pointer* to `opts` in both places.

See ``example1d1.cpp``, in the ``examples`` directory, for a simple
full working example.
Then to compile, linking to the double-precision static library, use eg::

  g++ example1d1.cpp -o example1d1 -I FINUFFT/include FINUFFT/lib-static/libfinufft.a -fopenmp -lfftw3_omp -lfftw3 -lm

where ``FINUFFT`` denotes the library location (top directory).
The ``examples`` and ``test`` directories are good places to see further
usage examples. The documentation for all 18 simple interfaces,
and the more powerful guru interface, follows below.

.. note::
 If you have a small-scale 2D task (say less than 10\ :sup:`5` points or modes) with multiple strength or coefficient vectors but fixed nonuniform points, see the :ref:`advanced interfaces <advinterface>`.


Complete documentation for C/C++
-------------------------------- 
 
 .. _datatypes:
 
Data types
~~~~~~~~~~


There are certain data type names
that we found convenient to unify the interfaces. These are used throughout
the below.

- ``FLT`` : this means ``double`` if compiled in
  the default double-precision, or ``float`` if compiled in single precision.
  This is used for all real-valued input and output arrays.

- ``CPX`` : means ``complex<double>`` in double precision,
  or ``complex<float>`` in single precision.
  This is used for all complex-valued input and output arrays.
  In the documentation this is often referred to as ``complex FLT``.

- ``BIGINT`` : this is the signed integer type used for all potentially-large input arguments, such as ``M`` and ``N`` in the example above. It is defined to the signed 64-bit integer type ``int64_t``, allowing the number of input points and/or output modes to exceed 2^31 (around 2 billion). Internally, the ``BIGINT`` type is also used for all relevant indexing; we have not noticed a slow-down relative to using 32-bit integers (the advanced user could explore this by changing its definition in ``finufft.h`` and recompiling).
  This is also referred to as ``int64`` in the documentation.

- ``int`` : (in contrast to the above)
  is the usual 32-bit signed integer, and is used for
  flags (such as the value ``+1`` used above) and the output error code.


  
Here we describe the simple interfaces to call FINUFFT from C++, C.

We provide Type 1 (nonuniform to uniform), Type 2 (uniform to
nonuniform), and Type 3 (nonuniform to nonuniform), in dimensions 1,
2, and 3.  This gives nine basic routines.
There are also two :ref:`advanced interfaces <advinterface>`
for multiple 2d1 and 2d2 transforms with the same point locations.

         *** TO DISCUSS! UPDATE ! ********

         
Using the library is a matter of filling your input arrays,
allocating the correct output array size, possibly setting fields in
the options struct, then calling one of the transform routines below.

.. warning::
   FINUFFT (when compiled with OpenMP) by default uses all available threads,
   which is often twice the number of cores (full hyperthreading).
   We have observed that a large thread
   count can lead to *reduced* performance, presumably because RAM access is the limiting factor. We recommend that one limit the
   number of threads at most around 24. This can be done in linux via
   the shell environment, eg ``OMP_NUM_THREADS=16``, or using OpenMP
   commands in the various languages.

   

Interfaces from C++
*******************


  

1D transforms, simple interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we list the calling sequences for the main C++ codes.
Please refer to the above :ref:`data types <datatypes>`.

::

  int finufft1d1(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
                 CPX* fk, nufft_opts opts)
   
   Type-1 1D complex nonuniform FFT.

              nj-1
     fk(k1) = SUM cj[j] exp(+/-i k1 xj(j))  for -ms/2 <= k1 <= (ms-1)/2
              j=0                            
   Inputs:
     nj     number of sources (int64, aka BIGINT)
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
            stored as alternating Re & Im parts (2*ms FLTs),
 	    order determined by opts.modeord.
     returned value - 0 if success, else see ../docs/usage.rst


   
  int finufft1d2(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
                 CPX* fk, nufft_opts opts)
  
   Type-2 1D complex nonuniform FFT.

     cj[j] = SUM   fk[k1] exp(+/-i k1 xj[j])      for j = 0,...,nj-1
             k1 
     where sum is over -ms/2 <= k1 <= (ms-1)/2.

   Inputs:
     nj     number of targets (int64, aka BIGINT)
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



  int finufft1d3(BIGINT nj,FLT* xj,CPX* cj,int iflag, FLT eps, BIGINT nk,
                 FLT* s, CPX* fk, nufft_opts opts)
  
   Type-3 1D complex nonuniform FFT.

               nj-1
     fk[k]  =  SUM   c[j] exp(+-i s[k] xj[j]),      for k = 0, ..., nk-1
               j=0
   Inputs:
     nj     number of sources (int64, aka BIGINT)
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

     

2D transforms, simple interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

  int finufft2d1(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,
	       FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts)

   Type-1 2D complex nonuniform FFT.

                  nj-1
     f[k1,k2] =   SUM  c[j] exp(+-i (k1 x[j] + k2 y[j]))
                  j=0

     for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2.

     The output array is k1 (fast), then k2 (slow), with each dimension
     determined by opts.modeord.
     If iflag>0 the + sign is used, otherwise the - sign is used,
     in the exponential.

   Inputs:
     nj     number of sources (int64, aka BIGINT)
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



  int finufft2d2(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts)

   Type-2 2D complex nonuniform FFT.

     cj[j] =  SUM   fk[k1,k2] exp(+/-i (k1 xj[j] + k2 yj[j]))      for j = 0,...,nj-1
             k1,k2
     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2,

   Inputs:
     nj     number of targets (int64, aka BIGINT)
     xj,yj     x,y locations of targets (each a size-nj FLT array) in [-3pi,3pi]
     fk     FLT complex array of Fourier transform values (size ms*mt,
            increasing fast in ms then slow in mt, ie Fortran ordering).
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

     

  int finufft2d3(BIGINT nj,FLT* xj,FLT* yj,CPX* cj,int iflag, FLT eps,
                 BIGINT nk, FLT* s, FLT *t, CPX* fk, nufft_opts opts)

   Type-3 2D complex nonuniform FFT.

               nj-1
     fk[k]  =  SUM   c[j] exp(+-i (s[k] xj[j] + t[k] yj[j]),    for k=0,...,nk-1
               j=0
   Inputs:
     nj     number of sources (int64, aka BIGINT)
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

   
3D transforms, simple interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

  int finufft3d1(BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,
	       FLT eps, BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk,
	       nufft_opts opts)

   Type-1 3D complex nonuniform FFT.

                     nj-1
     f[k1,k2,k3] =   SUM  c[j] exp(+-i (k1 x[j] + k2 y[j] + k3 z[j]))
                     j=0

	for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2,
            -mu/2 <= k3 <= (mu-1)/2.

     The output array is as in opt.modeord in each dimension.
     k1 changes is fastest, k2 middle,
     and k3 slowest, ie Fortran ordering. If iflag>0 the + sign is
     used, otherwise the - sign is used, in the exponential.
                           
   Inputs:
     nj     number of sources (int64, aka BIGINT)
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


     
  int finufft3d2(BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,
	       int iflag,FLT eps, BIGINT ms, BIGINT mt, BIGINT mu,
	       CPX* fk, nufft_opts opts)

   Type-2 3D complex nonuniform FFT.

     cj[j] =    SUM   fk[k1,k2,k3] exp(+/-i (k1 xj[j] + k2 yj[j] + k3 zj[j]))
             k1,k2,k3
      for j = 0,...,nj-1
     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2, 
                       -mu/2 <= k3 <= (mu-1)/2

   Inputs:
     nj     number of sources (int64, aka BIGINT)
     xj,yj,zj     x,y,z locations of targets (each size-nj FLT array) in [-3pi,3pi]
     fk     FLT complex array of Fourier series values (size ms*mt*mu,
            increasing fastest in ms to slowest in mu, ie Fortran ordering).
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



  int finufft3d3(BIGINT nj,FLT* xj,FLT* yj,FLT *zj, CPX* cj,
	       int iflag, FLT eps, BIGINT nk, FLT* s, FLT *t,
	       FLT *u, CPX* fk, nufft_opts opts)

   Type-3 3D complex nonuniform FFT.

               nj-1
     fk[k]  =  SUM   c[j] exp(+-i (s[k] xj[j] + t[k] yj[j] + u[k] zj[j]),
               j=0
                          for k=0,...,nk-1
   Inputs:
     nj     number of sources (int64, aka BIGINT)
     xj,yj,zj   x,y,z location of sources in R^3 (each size-nj FLT array)
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

  
Interfaces from C
*****************

From C one calls the same routines as for C++, and includes
the same header files (this unified interface is new as of version 1.1).
To recap, one should ``#include "finufft.h"`` then, as above, initialize the options:

  nufft_opts opts; finufft_default_opts(&opts);

Options fields may then be changed in ``opts`` before calling ``finufft?d?``
(where the wildcard ``?`` denotes an appropriate number).

As above, ``FLT`` indicates ``double`` or ``float``, but now
``CPX`` indicates their complex C99-type equivalents
(see ``src/finufft.h`` for the definitions used).
For examples see ``examples/example1d1c.c`` (double precision)
and ``examples/example1d1cf.c`` (single precision).


Usage and design notes
**********************

- We strongly recommend you use ``upsampfac=1.25`` for type-3; it
  reduces its run-time from around 8 times the types 1 or 2, to around 3-4
  times. It is often also faster for type-1 and type-2, at low precisions.

- Sizes >=2^31 have been tested for C++ drivers (``test/finufft?d_test.cpp``), and
  work fine, if you have enough RAM.
  In fortran the interface is still 32-bit integers, limiting to
  array sizes <2^31. The fortran interface needs to be improved.

- C++ is used for all main libraries, almost entirely avoiding object-oriented code. C++ ``std::complex<double>`` (typedef'ed to ``CPX`` and sometimes ``dcomplex``) and FFTW complex types are mixed within the library, since to some extent our library is a glorified driver for FFTW. FFTW was considered universal and essential enough to be a dependency for the whole package.

- There is a hard-defined limit of ``1e11`` for the size of internal FFT arrays, set in ``defs.h`` as ``MAX_NF``: if your machine has RAM of order 1TB, and you need it, set this larger and recompile. The point of this is to catch ridiculous-sized mallocs and exit gracefully. Note that mallocs smaller than this, but which still exceed available RAM, cause segfaults as usual. For simplicity of code, we do not do error checking on every malloc.

- As a spreading kernel function, we use a new faster simplification of the Kaiser--Bessel kernel, and eventually settled on piecewise polynomial approximation of this kernel.  At high requested precisions, like the Kaiser--Bessel, this achieves roughly half the kernel width achievable by a truncated Gaussian. Our kernel is exp(-beta.sqrt(1-(2x/W)^2)), where W = nspread is the full kernel width in grid units. This (and Kaiser--Bessel) are good approximations to the prolate spheroidal wavefunction of order zero (PSWF), being the functions of given support [-W/2,W/2] whose Fourier transform has minimal L2 norm outside of a symmetric interval. The PSWF frequency parameter (see [ORZ]) is c = pi.(1-1/2sigma).W where sigma is the upsampling parameter. See our paper in the references.
