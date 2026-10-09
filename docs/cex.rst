.. _cex:

Example usage from C++ and C
=================================

.. _quick:

Quick-start example in C++
--------------------------

Here's how to perform a 1D type-1 transform
in double precision from C++, using STL complex vectors.
From the repository root, build and run the first example::

  cmake -S . -B build -DFINUFFT_BUILD_EXAMPLES=ON
  cmake --build build
  build/examples/simple1d1

First include our header, and some others needed for the demo:

.. literalinclude:: ../examples/simple1d1.cpp
  :language: C++
  :start-after: docs-start: quick-start
  :end-before: docs-end: quick-start

We need nonuniform points ``x`` and complex strengths ``c``. Let's create random ones for now,
drawn from a fixed-seed generator (``mt19937`` + ``uniform_real_distribution``):

.. literalinclude:: ../examples/simple1d1.cpp
  :language: C++
  :start-after: docs-start: walkthrough
  :end-before: docs-end: walkthrough

The full example file is ``examples/simple1d1.cpp``. Build and run all examples from the repository root with ``cmake -S . -B build -DFINUFFT_BUILD_EXAMPLES=ON`` and ``cmake --build build``.

Now do the NUFFT (with default options; here we pass our own, initialized to defaults). Since the interface is
C-compatible, we pass pointers to the start of the arrays (rather than
C++-style vector objects), and also pass ``N``:

.. literalinclude:: ../examples/simple1d1.cpp
  :language: C++
  :start-after: docs-start: transform
  :end-before: docs-end: transform

The full example file is ``examples/simple1d1.cpp``. Compile it against the static library with ``g++ -fopenmp simple1d1.cpp -o simple1d1 -I../include -Wl,--start-group ../build/src/libfinufft.a ../build/src/common/libfinufft_common.a -Wl,--end-group -lfftw3_omp -lfftw3 -lfftw3f_omp -lfftw3f``, or see ``examples/README`` for other linking options.

This fills ``F`` with the output modes, in increasing ordering
with the integer frequency indices from ``-N/2`` up to ``N/2-1``
(since ``N`` is even; for odd ``N`` it would be ``-(N-1)/2`` up to ``(N-1)/2``).
The transform (:math:`10^7` points to :math:`10^6` modes) takes 0.4 seconds on a laptop.
The index is thus offset by ``N/2`` (this is integer division in the odd case), so that frequency ``k`` is output in
``F[N/2 + k]``.
Here ``+1`` sets the sign of :math:`i` in the exponentials
(see :ref:`definitions <math>`),
``1e-9`` requests 9-digit relative tolerance, and ``ier`` is a status output
which is zero if successful (otherwise see :ref:`error codes <error>`).

.. note::

   FINUFFT works with a periodicity of :math:`2\pi` for type 1 and 2 transforms; see :ref:`definitions <math>`. For example, nonuniform points :math:`x=\pm\pi` are equivalent. The input points can be any real numbers: each coordinate is folded internally into :math:`[-\pi,\pi)`, so round-off error grows with :math:`|x|`. To use a different periodicity, linearly rescale your coordinates.

If instead you want to change some options, put default values in the
``finufft_opts`` struct, make your changes, then pass the pointer to FINUFFT.
For instance, to print timing/debug info, the call sequence differs from the
above only in::

  opts.debug = 1;                                // prints timing/debug info
  int ier = finufft1d1(M,&x[0],&c[0],+1,tol,N,&F[0],&opts);

.. warning::
   - Without the ``finufft_default_opts`` call, options may take on arbitrary values which may cause a crash.

See ``examples/simple1d1.cpp`` for a simple full working demo of the above, including a test of the math (the demo uses ``M=10^7`` and ``N=10^6``). If you instead use single-precision arrays,
replace the tag ``finufft`` by ``finufftf`` in each command; see ``examples/simple1d1f.cpp``.

From the ``examples/`` directory, to compile the C quick-start example on a Linux/GCC system, linking to the static library, use::

  gcc -fopenmp simple1d1c.c -o simple1d1c -I../include -Wl,--start-group ../build/src/libfinufft.a ../build/src/common/libfinufft_common.a -Wl,--end-group -lfftw3_omp -lfftw3 -lfftw3f_omp -lfftw3f -lstdc++ -lm

The ``-lstdc++`` is needed for any C code linking against FINUFFT; see ``examples/README`` for general compilation instructions for the examples.
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
languages. Here is a 2D type-1 example (excerpt; the full file with the final
accuracy check is ``examples/simple2d1.cpp``; it uses ``N1=2000``, ``N2=500``
and requests ``tol=1e-6``):

.. literalinclude:: ../examples/simple2d1.cpp
  :language: C++
  :start-after: docs-start: simple2d1
  :end-before: docs-end: simple2d1

The full example file is ``examples/simple2d1.cpp``. Compile it against the static library with ``g++ -fopenmp simple2d1.cpp -o simple2d1 -I../include -Wl,--start-group ../build/src/libfinufft.a ../build/src/common/libfinufft_common.a -Wl,--end-group -lfftw3_omp -lfftw3 -lfftw3f_omp -lfftw3f``, or see ``examples/README`` for other linking options.

This transform takes 0.6 seconds on a laptop.
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

See ``examples/simple2d1.cpp`` for the same example with a math check, to
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
Here we show how to do a stack of 1D type 1 NUFFT transforms, in C++
(excerpt; the full file with the final accuracy check is
``examples/many1d1.cpp``, which uses ``ntrans=3``).
The strength data vectors are taken to be contiguous (the whole
first vector, followed by the second, etc, rather than interleaved.)
Ie, viewed as a matrix in Fortran storage, each column is a strength vector.

.. literalinclude:: ../examples/many1d1.cpp
  :language: C++
  :start-after: docs-start: many1d1
  :end-before: docs-end: many1d1

The full example file is ``examples/many1d1.cpp``. Compile it against the static library with ``g++ -fopenmp many1d1.cpp -o many1d1 -I../include -Wl,--start-group ../build/src/libfinufft.a ../build/src/common/libfinufft_common.a -Wl,--end-group -lfftw3_omp -lfftw3 -lfftw3f_omp -lfftw3f``, or see ``examples/README`` for other linking options.

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
Here's the 2D type 1 C++ guru example (excerpt; the full file with the final
accuracy check is ``examples/guru2d1.cpp``; it uses ``N1=2000``, ``N2=500``
and requests ``tol=1e-6``):

.. literalinclude:: ../examples/guru2d1.cpp
  :language: C++
  :start-after: docs-start: guru2d1
  :end-before: docs-end: guru2d1

The full example file is ``examples/guru2d1.cpp``. Compile it against the static library with ``g++ -fopenmp guru2d1.cpp -o guru2d1 -I../include -Wl,--start-group ../build/src/libfinufft.a ../build/src/common/libfinufft_common.a -Wl,--end-group -lfftw3_omp -lfftw3 -lfftw3f_omp -lfftw3f``, or see ``examples/README`` for other linking options.

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
the following, which are built with ``-DFINUFFT_BUILD_EXAMPLES=ON``:

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
