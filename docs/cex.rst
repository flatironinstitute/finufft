.. _cex:

Example usage from C++ and C
=================================

.. _quick:

Quick-start example in C++
--------------------------

Here's how to perform a 1D type-1 transform
in double precision from C++, using STL complex vectors.
First include our header, and some others needed for the demo:

.. literalinclude:: ../examples/simple1d1.cpp
  :language: C++
  :start-after: docs-start: simple1d1-includes
  :end-before: docs-end: simple1d1-includes

We need nonuniform points ``x`` and complex strengths ``c``. Let's create random ones for now:

.. literalinclude:: ../examples/simple1d1.cpp
  :language: C++
  :dedent: 2
  :start-after: docs-start: simple1d1-data
  :end-before: docs-end: simple1d1-data

With ``N`` as the desired number of Fourier mode coefficients,
allocate their output array:

.. literalinclude:: ../examples/simple1d1.cpp
  :language: C++
  :dedent: 2
  :start-after: docs-start: simple1d1-alloc
  :end-before: docs-end: simple1d1-alloc

Now do the NUFFT (with default options, indicated by the ``nullptr`` in the following call). Since the interface is
C-compatible, we pass pointers to the start of the arrays (rather than
C++-style vector objects), and also pass ``N``:

.. literalinclude:: ../examples/simple1d1.cpp
  :language: C++
  :dedent: 2
  :start-after: docs-start: simple1d1-call
  :end-before: docs-end: simple1d1-call

This fills ``F`` with the output modes, in increasing ordering
with the integer frequency indices from ``-N/2`` up to ``N/2-1``
(since ``N`` is even; for odd it would be ``-(N-1)/2`` up to ``(N-1)/2``).
The index is thus offset by ``N/2`` (this is integer division in the odd case), so that frequency ``k`` is output in
``F[N/2 + k]``.
Here ``+1`` sets the sign of :math:`i` in the exponentials
(see :ref:`definitions <math>`),
``acc=1e-9`` requests 9-digit relative tolerance, and ``ier`` is a status output
which is zero if successful (otherwise see :ref:`error codes <error>`).
With :math:`10^7` points and :math:`10^6` modes, as above, the transform takes 0.2 seconds on a laptop.
.. to generate that timing claim, since the examples do not have timing, use:   test/finufft1d_test 1e6 1e7 1e-9 0 2 0.0

.. note::

   FINUFFT works with a periodicity of :math:`2\pi` for type 1 and 2 transforms; see :ref:`definitions <math>`. For example, nonuniform points :math:`x=\pm\pi` are equivalent. The input points can be any real numbers: each coordinate is folded internally into :math:`[-\pi,\pi)`, so round-off error grows with :math:`|x|`. To use a different periodicity, linearly rescale your coordinates.

If instead you want to change some options, first
put default values in a ``finufft_opts`` struct,
make your changes, then pass the pointer to FINUFFT:

.. code-block:: C++

  finufft_opts opts;
  finufft_default_opts(&opts);
  opts.debug = 1;                                 // prints timing/debug info
  int ier = finufft1d1(M,x.data(),c.data(),+1,tol,N,F.data(),&opts);

.. warning::
   - Without the ``finufft_default_opts`` call, options may take on arbitrary values which may cause a crash.
   - Note that, as of version 2.0, a pointer to ``opts`` is passed in both places.

See ``examples/simple1d1.cpp`` for a simple full working demo of the above, including a test of the math. If you instead use single-precision arrays,
replace the tag ``finufft`` by ``finufftf`` in each command; see ``examples/simple1d1f.cpp``.

From the ``examples/`` directory, to compile on a linux/GCC system, linking to the static library, use eg::

  g++ -std=c++17 -fopenmp simple1d1.cpp -o simple1d1 -I../include ../lib-static/libfinufft.a -lfftw3_omp -lfftw3 -lfftw3f_omp -lfftw3f

Executing ``./simple1d1`` should now work (exit code ``0`` and displaying a small error).
If you used ``FFT=DUCC`` you can of course drop the linking of the four ``fftw3`` libraries.
Better is instead to link to the dynamic shared (``.so``) library, via eg::

  g++ -std=c++17 -fopenmp simple1d1.cpp -o simple1d1 -I../include -Wl,-rpath,$FINUFFT/lib/ -lfinufft

where ``$FINUFFT`` must be replaced by (or be an environment variable set to) the absolute install path for this repository.
Notice how ``rpath`` is used to make an executable that may be called from, or moved to, anywhere.
See ``examples/README`` for general compilation instructions for the examples.
The ``examples`` and ``test`` directories are good places to see further
usage examples. The documentation for all 18 simple interfaces,
and the more flexible guru interface, is further down this page.

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
  finufft_opts opts;                      // make an opts struct
  finufft_default_opts(&opts);          // set default opts (must do this)
  opts.debug = 2;                       // more debug/timing to stdout
  int ier = finufft1d1(M,x,c,+1,1e-9,N,F,&opts);

  // (now do something with F here!...)

  free(x); free(c); free(F);

See ``examples/simple1d1c.c`` and ``examples/simple1d1cf.c`` for
double- and single-precision C examples, including the math check to insure
the correct indexing of output modes. Don't forget to compile your C code with
``-lstdc++`` when linking against FINUFFT.


2D example in C++
-----------------

We assume Fortran-style contiguous multidimensional arrays, as opposed
to C-style arrays of pointers; this allows the widest compatibility with other
languages. Assuming the same headers as above, we first create points
:math:`(x_j,y_j)` in the square :math:`[-\pi,\pi)^2`, and strengths as before
(each component uniform random in :math:`[-1,1]`):

.. literalinclude:: ../examples/simple2d1.cpp
  :language: C++
  :dedent: 2
  :start-after: docs-start: simple2d1-data
  :end-before: docs-end: simple2d1-data

Let's say we want to compute ``N1=1000`` by ``N2=2000``
2D Fourier coefficients. We allocate the output array:

.. literalinclude:: ../examples/simple2d1.cpp
  :language: C++
  :dedent: 2
  :start-after: docs-start: simple2d1-alloc
  :end-before: docs-end: simple2d1-alloc

and do the transform (here with ``upsampfac=1.25``) thus:

.. literalinclude:: ../examples/simple2d1.cpp
  :language: C++
  :dedent: 2
  :start-after: docs-start: simple2d1-call
  :end-before: docs-end: simple2d1-call

This transform (from 10 million points to 2 million modes)
takes 0.3 seconds on a laptop.
.. to generate that timing claim, use:   est/finufft2d_test 1000 2000 1e7 1e-6 0 2 0
The modes have increasing ordering
of integer frequency indices from ``-N1/2`` up to ``N1/2-1``
in the fast (``x``) dimension,
then indices from ``-N2/2`` up to ``N2/2-1`` in the slow (``y``) dimension
(since both ``N1`` and ``N2`` are even).
So, the output frequency ``(k1,k2)`` is found in
``F[N1/2 + k1 + (N2/2 + k2)*N1]``.

See ``opts.modeord`` in :ref:`Options<opts>`
to instead use FFT-style mode ordering, which
simply differs by an "fftshift" (as it is commonly called).

See ``examples/simple2d1.cpp`` for an example with a math check, to
insure that the mode indexing is correctly understood.


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
Here we show how to do a stack of ``ntrans=10`` 1D type 1 NUFFT transforms
each similar to the first example above, in C++,
assuming the same headers as in that example.
The strength data vectors are taken to be contiguous (the whole
first vector, followed by the second, etc, rather than interleaved.)
Ie, viewed as a matrix in Fortran storage, each column is a strength vector.

First create the ``M`` points ``x``, and ``M*ntrans`` strengths ``c`` filling all ``ntrans`` vectors:

.. literalinclude:: ../examples/many1d1.cpp
  :language: C++
  :dedent: 2
  :start-after: docs-start: many1d1-data
  :end-before: docs-end: many1d1-data

Then allocate ``ntrans`` output vectors, ``N*ntrans`` modes in total:

.. literalinclude:: ../examples/many1d1.cpp
  :language: C++
  :dedent: 2
  :start-after: docs-start: many1d1-alloc
  :end-before: docs-end: many1d1-alloc

One call does the whole stack (with default options):

.. literalinclude:: ../examples/many1d1.cpp
  :language: C++
  :dedent: 2
  :start-after: docs-start: many1d1-call
  :end-before: docs-end: many1d1-call

The stack of transforms takes 0.3 seconds on a laptop, about twice as fast as making 10 separate "simple" calls.
.. to generate those timing claims, use:    test/finufft1dmany_test 10 1e6 2e6 1e-9 0 0 2 0.0
The frequency index ``k`` in transform number ``t`` (zero-indexing the transforms) is in ``F[k + (int)N/2 + N*t]``.

See ``examples/many1d1.cpp`` and ``test/finufft?dmany_test.cpp``
for more examples.


Guru interface examples
-----------------------

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
Finally, you may execute *adjoints* of the planned transforms without
re-planning, making forward-adjoint transform pairs very convenient.
Now we redo the above 2D type 1 C++ example with the guru interface.

Assume ``x``, ``y``, ``c`` are filled, and ``F`` allocated, as in the 2D code above.
Step 1: one first makes a plan giving transform parameters, but no data:

.. literalinclude:: ../examples/guru2d1.cpp
  :language: C++
  :dedent: 2
  :start-after: docs-start: guru2d1-makeplan
  :end-before: docs-end: guru2d1-makeplan

Step 2: send in pointers to the ``M`` nonuniform points (just ``x``, ``y`` in this case):

.. literalinclude:: ../examples/guru2d1.cpp
  :language: C++
  :dedent: 2
  :start-after: docs-start: guru2d1-setpts
  :end-before: docs-end: guru2d1-setpts

The user should not change the ``x``, ``y`` nonuniform point arrays from here on.
Step 3: do the planned transform to the ``c`` strength data, output to ``F``:

.. literalinclude:: ../examples/guru2d1.cpp
  :language: C++
  :dedent: 2
  :start-after: docs-start: guru2d1-execute
  :end-before: docs-end: guru2d1-execute

You could now send in new points, and/or do transforms with new ``c`` data,
or even adjoint transforms with the same points but now mapping ``F`` to ``c``.
Step 4: when done, free the memory used by the plan:

.. literalinclude:: ../examples/guru2d1.cpp
  :language: C++
  :dedent: 2
  :start-after: docs-start: guru2d1-destroy
  :end-before: docs-end: guru2d1-destroy

This writes the Fourier coefficients to ``F`` just as in the earlier 2D example.
One difference from the above simple and vectorized interfaces
is that the ``int64_t`` type (aka ``long long int``)
is needed since the Fourier coefficient dimensions are passed as an array.

.. warning::
  You must not change the nonuniform point arrays (here ``x``, ``y``) between passing them to ``finufft_setpts`` and performing ``finufft_execute`` or ``finufft_execute_adjoint``. The last two calls expect these arrays to be unchanged. We chose this style of interface since it saves RAM and time (by avoiding unnecessary duplication), allowing the largest possible problems to be solved.

.. warning::
  You must destroy a plan before making a new plan using the same
  plan object, otherwise a memory leak results.

The complete code with a math test is in ``examples/guru2d1.cpp``,
the demo of an adjoint execution is in ``examples/guru2d1_adjoint.cpp``,
and for more examples see ``examples/guru1d1*.c*``

Using the guru interface to perform a vectorized transform (multiple 1D type 1
transforms each with the same nonuniform points) is demonstrated in
``examples/gurumany1d1.cpp``. This is similar to the single-command vectorized
interface, but allowing more control (changing the nonuniform points without
re-planning the FFT, etc).


Thread safety for single-threaded transforms, and global state
--------------------------------------------------------------

It is possible to call FINUFFT from within multithreaded code, e.g. in an
OpenMP parallel block. In this case ``opts.nthreads=1`` should be set, otherwise
a segfault will occur. This is useful if you don't want to synchronize
independent transforms.
For demos of this "parallelize over single-threaded transforms" use case, see
the following, which are built as part of the ``make examples`` task:

* ``examples/threadsafe1d1`` which runs a 1D type-1 separately on each thread, checking the math, and

* ``examples/threadsafe2d2f`` which runs a 2D type-2 on each "slice" (in the MRI
  language), parallelized over slices with an OpenMP parallel for loop.
  (In this code there is no math check, just status check.)

However, if you have multiple transforms with the *same* nonuniform points for
each transform, it is probably much faster to use the vectorized interface,
and do all these transforms with a single such multithreaded FINUFFT call
(see ``examples/many1d1.cpp`` and ``examples/gurumany1d1.cpp``).
This may be less convenient if you want to leave your slices unsynchronized.

.. note::
   A design decision of FFTW is to have a global state which stores
   wisdom and settings. Such global state can cause unforeseen effects on other
   routines that also use FFTW. In contrast, FINUFFT uses pointers to plans to store
   its state, and does not have a global state (other than one ``static``
   flag used as a lock on FFTW initialization in the FINUFFT plan
   stage). This means different FINUFFT calls should not affect each other,
   although they may affect other codes that use FFTW via FFTW's global state.
