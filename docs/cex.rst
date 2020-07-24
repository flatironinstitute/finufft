.. _cex:

Example usage from C++ and C
=================================

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
from frequency index ``-N/2`` up to ``N/2-1``. The transform (:math:`10^7` points to :math:`10^6` modes) takes 0.4 seconds on a laptop.
The indexing is offset by ``(int)N/2``, so that frequency ``k`` is output in
``F[(int)N/2 + k]``.
Here ``+1`` sets the sign of :math:`i` in the exponentials
(see :ref:`definitions <math>`),
``1e-9`` requests 9-digit relative tolerance, and ``ier`` is a status output
which is zero if successful (otherwise see :ref:`error codes <error>`).

.. note::

   FINUFFT works with a periodicity of :math:`2\pi` for type 1 and 2 transforms; see :ref:`definitions <math>`. For example, nonuniform points :math:`x=\pm\pi` are equivalent. Points must lie in the input domain :math:`[-3\pi,3\pi)`, which allows the user to assume a convenient periodic domain such as  :math:`[-\pi,\pi)` or :math:`[0,2\pi)`. To handle points outside of :math:`[-3\pi,3\pi)` the user must fold them back into this domain before passing to FINUFFT. FINUFFT does not handle this case, for speed reasons. To use a different periodicity, linearly rescale your coordinates.

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
   - This usage is new as of version 1.2: ``opts`` is passed as a pointer in both places.

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
--------------------------

The FINUFFT C++ interface is intentionally also C-compatible, for simplity.
Thus, to use from C, the above example only needs to replace the C++
``vector`` with C-style array creation. Using C99 style, the
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
                
  // (now do something with F here!...)
                
  free(x); free(c); free(F);
                
See ``examples/simple1d1c.c`` and ``examples/simple1d1cf.c`` for
double- and single-precision C examples, including the math check to insure
the correct indexing of output modes.


2D example in C++
-----------------

We assume Fortran-style contiguous multidimensional arrays, as opposed
to C-style arrays of pointers; this allows the widest compatibility with other
languages. Assuming the same headers as above, we first create points
:math:`(x_j,y_j)` in the square :math:`[-\pi,\pi)^2`, and strengths as before:

.. code-block:: C++

  int M = 1e7;                                   // number of nonuniform points
  vector<double> x(M), y(M);
  vector<complex<double> > c(M);
  for (int j=0; j<M; ++j) {
    x[j] = M_PI*(2*((double)rand()/RAND_MAX)-1);
    y[j] = M_PI*(2*((double)rand()/RAND_MAX)-1);
    c[j] = 2*((double)rand()/RAND_MAX)-1 + I*(2*((double)rand()/RAND_MAX)-1);
  }

Let's say we want ``N1=1000`` by ``N2=2000`` 2D Fourier coefficients.
We allocate and do the (default options) transform thus:

.. code-block:: C++

  int N1=1000, N2=2000;
  vector<complex<double> > F(N1*N2);
  int ier = finufft2d1(M,&x[0],&y[0], &c[0], +1, 1e-6, N1, N2, &F[0], NULL);

This transform takes 0.6 seconds on a laptop.
The modes have increasing ordering
from frequency index ``-N1/2`` to ``N1/2-1`` in the fast (``x``) dimension,
then ordering ``-N2/2`` up to ``N2/2-1`` in the slow (``y``) dimension.
So, the output frequency ``(k1,k2)`` is found in
``F[(int)N1/2 + k1 + ((int)N2/2 + k2)*N1]``.

See ``opts.modeord`` in :ref:`Options<opts>`
to instead use FFT-style mode ordering, which
simply differs by an ``fftshift`` (as it is commonly called).

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

This takes 2.6 seconds on a laptop, around 1.4x faster than
making 10 separate "simple" calls.
The frequency index ``k`` in transform number ``t`` (zero-indexing the transforms) is in ``F[k + (int)N/2 + N*t]``.

See ``examples/many1d1.cpp`` and ``test/finufft?dmany_test.cpp``
for more examples.


Guru interface example
----------------------

If you want more flexibility than the above, use the "guru" interface:
this is similar to that of FFTW3, and to the main interface of
`NFFT3 <https://www-user.tu-chemnitz.de/~potts/nfft/>`_.
It lets you change the nonuniform points while keeping the
same pointer to an FFTW plan for a particular number of stacked transforms
with a certain number of modes.
This avoids the overhead (typically 0.1 ms per thread) of FFTW checking for
previous wisdom which would be significant when doing many small transforms.
You may also send in a new
set of stacked strength data (for type 1 and 3, or coefficients for type 2),
reusing the existing FFTW plan and sorted points.
Now we redo the above 2D type 1 C++ example with the guru interface.

One first makes a plan giving transform parameters, but no data:

.. code-block:: C++

  // (assume x, y, c are filled, and F allocated, as in the 2D code above...)
  int type=1, dim=2, ntrans=1;
  int64_t Ns[] = {1000,2000};                    // N1,N2 as 64-bit int array
  // step 1: make a plan...
  finufft_plan plan;
  int ier = finufft_makeplan(type, dim, Ns, +1, ntrans, 1e-6, &plan, NULL);
  // step 2: send in M nonuniform points (just x, y in this case)...
  finufft_setpts(plan, M, &x[0], &y[0], NULL, 0, NULL, NULL, NULL);
  // step 3: do the planned transform to the c strength data, output to F...
  finufft_execute(plan, &c[0], &F[0]);
  // ... you could now send in new points, and/or do transforms with new c data
  // ...
  // step 4: when done, free the memory used by the plan...
  finufft_destroy(plan);

This writes the Fourier coefficients to ``F`` just as in the earlier 2D example.
One difference from the above simple and vectorized interfaces
is that the ``int64_t`` type (aka ``long long int``)
is needed since the Fourier coefficient dimensions are passed as an array.

  .. warning::
  You must destroy a plan before making a new plan using the same
  plan object, otherwise a memory leak results.

The complete code with a math test is in ``examples/guru2d1.cpp``, and for
more examples see ``examples/guru1d1*.c*``