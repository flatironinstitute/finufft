.. _c:

Usage from C++ and C
======================

.. _quick:

Quick-start example in C++
--------------------------

Here's how to perform a 1D type-1 transform
in double precision from C++, using STL complex vectors.
First include our header, and some others needed for the demo:

.. code-block:: C++
  
  #include "finufft.h"
  #include <vector>
  #include <complex>
  #include <stdlib.h>

We need nonuniform points ``x`` and complex strengths ``c``. Let's create random ones for now:
  
.. code-block:: C++

  int M = 1e7;                                   // number of nonuniform points
  vector<double> x(M);
  vector<complex<double> > c(M);
  complex<double> I = complex<double>(0.0,1.0);  // the imaginary unit
  for (int j=0; j<M; ++j) {
    x[j] = M_PI*(2*((double)rand()/RAND_MAX)-1); // uniform random in [-pi,pi)
    c[j] = 2*((double)rand()/RAND_MAX)-1 + I*(2*((double)rand()/RAND_MAX)-1);
  }

With ``N`` as the desired number of Fourier mode coefficients,
allocate their output array:

.. code-block:: C++
  
  int N = 1e6;                                   // number of output modes
  vector<complex<double> > F(N);

Now do the NUFFT (with default options, indicated by the ``NULL`` in the following call). Since the interface is
C-compatible, we pass pointers to the start of the arrays (rather than
C++-style vector objects), and also pass ``N``:

.. code-block:: C++

  int ier = finufft1d1(M,&x[0],&c[0],+1,1e-9,N,&F[0],NULL);

This fills ``F`` with the output modes, in increasing ordering
from frequency index ``-N/2`` up to ``N/2-1``. Transforming :math:`10^7` points
to :math:`10^6` modes takes 1-2 seconds on a laptop.
The indexing is offset by ``(int)N/2``, so that frequency ``k`` is output in
``F[(int)N/2 + k]``.
Here ``+1`` sets the sign of :math:`i` in the exponentials
(see :ref:`definitions <math>`),
``1e-9`` requests 9-digit relative tolerance, and ``ier`` is a status output
which is zero if successful (otherwise see :ref:`error codes <error>`).

.. note::

   FINUFFT works with a periodicity of :math:`2\pi` for type 1 and 2 transforms; see :ref:`definitions <math>`. For example, nonuniform points :math:`x=\pm\pi` are equivalent. Points must lie in the input domain :math:`[-3\pi,3\pi)`, which allows the user to assume a convenient periodic domain such as  :math:`[-\pi,\pi)` or :math:`[0,2\pi)`. To handle points outside this input domain, the user must fold them back into the input domain (FINUFFT does not do this for reasons of speed). To use a different periodicity, linearly rescale your coordinates.

If instead you want to change some options, first
put default values in a ``nufft_opts`` struct,
make your changes, then pass the pointer to FINUFFT:

.. code-block:: C++
  
  nufft_opts* opts = new nufft_opts;
  finufft_default_opts(opts);
  opts->debug = 1;                                // prints timing/debug info
  int ier = finufft1d1(M,&x[0],&c[0],+1,tol,N,&F[0],opts);
  
.. warning::
   - Without the ``finufft_default_opts`` call, options may take on arbitrary values which may cause a crash.
   - This usage is new as of version 1.2: `opts` is a pointer that is passed in both places.

See ``examples/simple1d1.cpp`` for a simple full working demo of the above, including a test of the math. If you instead use single-precision arrays,
replace the tag ``finufft`` by ``finufftf`` in each command; see ``examples/simple1d1f.cpp``.

Then to compile on a linux/GCC system, linking to the double-precision static library, use eg::

  g++ simple1d1.cpp -o simple1d1 -I$FINUFFT/include $FINUFFT/lib-static/libfinufft.a -fopenmp -lfftw3_omp -lfftw3 -lm

where ``$FINUFFT`` denotes the absolute path of your FINUFFT installation.
Better is instead link to dynamic shared (``.so``) libraries, via eg::

  g++ simple1d1.cpp -o simple1d1 -I$FINUFFT/include -L$FINUFFT/lib -lfinufft -lm
  
The ``examples`` and ``test`` directories are good places to see further
usage examples. The documentation for all 18 simple interfaces,
and the more flexible guru interface, follows below.

Quick-start example in C
-----------------------

FINUFFT is intentionally C-compatible.
Thus, to use from C, the above example only needs to replace the C++
``vector``s with C-style array creation. Using C99 style, the
above code, with options setting, becomes:

.. code-block:: C

#include <finufft.h>
#include <stdlib.h>
#include <complex.h>

  int M = 1e7;            // number of nonuniform points
  double* x = (double *)malloc(sizeof(double)*M);
  double complex* c = (double complex*)malloc(sizeof(double complex)*M);
  for (int j=0; j<M; ++j) {
    x[j] = M_PI*(2*((double)rand()/RAND_MAX)-1);  // uniform random in [-pi,pi)
    c[j] = 2*((double)rand()/RAND_MAX)-1 + I*(2*((double)rand()/RAND_MAX)-1);
  }
  int N = 1e6;            // number of modes
  double complex* F = (double complex*)malloc(sizeof(double complex)*N);
  nufft_opts opts;                      // make an opts struct
  finufft_default_opts(&opts);          // set default opts (must do this)
  opts.debug = 2;                       // more debug/timing to stdout
  int ier = finufft1d1(M,x,c,+1,1e-9,N,F,&opts);
  // (do something with F here!...)
  free(x); free(c); free(F);
                
See ``examples/simple1d1c.c`` and ``examples/simple1d1cf.c`` for
double- and single-precision C examples, including the math check to insure
the correct indexing of output modes.


Two-dimensional example in C++
------------------------------

We assume Fortran-style contiguous multidimensional arrays, as opposed
to C-style arrays of pointers; this allows the widest compatibility with other
languages. Assuming the same ``include``s as above, we first create points
:math:`(x_j,y_j)` in the square :math:`[-\pi,pi)^2`, and strengths as before:

.. code-block:: C++

  int M = 1e7;                                   // number of nonuniform points
  vector<double> x(M), y(M);
  vector<complex<double> > c(M);
  for (int j=0; j<M; ++j) {
    x[j] = M_PI*(2*((double)rand()/RAND_MAX)-1);
    y[j] = M_PI*(2*((double)rand()/RAND_MAX)-1);
    c[j] = 2*((double)rand()/RAND_MAX)-1 + I*(2*((double)rand()/RAND_MAX)-1);
  }

Let's say we want ``N1=1000`` by ``N2=500`` 2D Fourier coefficients.
We allocate and do the (default options) transform thus:

.. code-block:: C++

  int N1=1000, N2=500;
  vector<complex<double> > F(N1*N2);
  int ier = finufft2d1(M,&x[0],&y[0], &c[0], +1, 1e-6, N1, N2, &F[0], NULL);

The modes have increasing ordering
from frequency index ``-N1/2`` to ``N1/2-1`` in the fast (``x``) dimension,
then ordering ``-N2/2`` up to ``N2/2-1`` in the slow (``y``) dimension.
So, the output frequency ``(k1,k2)`` is found in
``F[(int)N1/2 + k1 + ((int)N2/2 + k2)*N1]``.

See ``opts.modeord`` to instead use FFT-style mode ordering, which
simply differs an ``fftshift`` (as it is commonly called).

See ``examples/simple2d1.cpp`` for an example with a math check, to
insure the modes are correctly indexed.


Vectorized interface example
----------------------------

A common use case is to perform a stack of identical transforms with the
same size and nonuniform points, but for new strength vectors.
(Applications include interpolating vector-valued data, or processing
MRI images collected with a fixed set of k-space sample points.)
Because it amortizes sorting, FFTW planning, and FFTW plan lookup,
it can be faster to use a "vectorized"
interface (which does the entire stack in one call)
than to repeatedly call the above "simple" interfaces.
This is especially true for many small problems.
Here we show how to do a stack of ``ntrans=10`` 1D type 1 NUFFT transforms, in C++,
assuming the same headers as in the first example above.
The strength data vectors are taken to be contiguous (the whole
first vector, followed by the second, etc, rather than interleaved.)
Ie, viewed as a matrix in Fortran storage, each column is a strength vector.

.. code-block:: C++

  int ntrans = 10;                               // how many transforms
  int M = 1e7;                                   // number of nonuniform points
  vector<double> x(M);
  vector<complex<double> > c(M*ntrans);          // ntrans strength vectors
  complex<double> I = complex<double>(0.0,1.0);  // the imaginary unit
  for (int j=0; j<M; ++j)
    x[j] = M_PI*(2*((double)rand()/RAND_MAX)-1);
  for (int j=0; j<M*ntrans; ++j)                 // fill all ntrans vectors...
    c[j] = 2*((double)rand()/RAND_MAX)-1 + I*(2*((double)rand()/RAND_MAX)-1);
  int N = 1e6;                                   // number of output modes
  vector<complex<double> > F(N*trans);           // ntrans output vectors
  int ier = finufft1d1(M,&x[0],&c[0],+1,1e-9,N,&F[0],NULL);    // default opts

This takes just over 5 seconds on a laptop, which is around 2.5x faster than
making 10 separate "simple" calls, quite an efficiency gain.
The frequency index ``k`` in the ``t``th transform (zero-indexing the transforms) is in ``F[k + (int)N/2 + N*t]``.

See ``examples/many1d1.cpp`` and ``test/finufft?dmany_test.cpp``
for more examples.


Guru interface example
----------------------

More flexible than the above interface is our "guru" interface;
this is modelled on that of FFTW3, and similar to the main interface of
`NFFT3 <https://www-user.tu-chemnitz.de/~potts/nfft/>`_.
This lets you change the nonuniform points while keeping the
same pointer to an FFTW plan for a particular number of stacked transforms
with a certain number of modes.
This avoids the overhead (typically 0.1 ms per thread) of FFTW checking for
previous wisdom, which can cause a huge slow-down for many small transforms.
You may also send in a new
set of stacked strength data (for type 1 and 3, or coefficients for type 2),
reusing the existing FFTW plan and sorted points.
Here is a 1D type 1 example in C++.

One first makes a plan giving transform parameters, but no data:

.. code-block:: C++

  ***


  finufft_makeplan(type, dim, Ns, +1, ntransf, tol, &plan, NULL);

*** 

  

.. note::
  User must destroy a plan before making a new plan using the same
  plan object,
  otherwise mem leak.






Simple interfaces
-------------------------------- 

FIX THE BELOW - REMOVE REPETITIONS:
(don't have to have each 18 interfaces listed out in full).


 .. _datatypes:
 
Data types
~~~~~~~~~~

We define data types that are convenient to unify the interfaces.
These are used throughout the below.

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

  



.. note::
 If you have a small-scale 2D task (say less than 10\ :sup:`5` points or modes) with multiple strength or coefficient vectors but fixed nonuniform points, see the :ref:`advanced interfaces <advinterface>`.










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
