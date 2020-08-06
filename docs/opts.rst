.. _opts:

Options parameters
==================

Aside from the mandatory inputs (dimension, type,
nonuniform points, strengths or coefficients, and, in C++/C/Fortran/MATLAB,
sign of the imaginary unit and tolerance)
FINUFFT has optional parameters.
These adjust the workings of the algorithm, change the output format,
or provide debug/timing text to stdout.
Sensible default options are chosen, so that the new user need not worry about
changing them.
However, users wanting to try to increase speed or see more
timing breakdowns will want to change options from their defaults.
See each language doc page for how this is done, but is generally
by creating an options structure, changing fields from their defaults,
then passing this (or a pointer to it)
to the simple, vectorized, or guru makeplan routines.
Recall how to do this from C++:

.. code-block:: C++
                
  // (... set up M,x,c,tol,N, and allocate F here...)
  nufft_opts* opts;
  finufft_default_opts(opts);
  opts->debug = 1;
  int ier = finufft1d1(M,x,c,+1,tol,N,F,opts);

This setting produces more timing output to ``stdout``.

.. warning::
   
 In C/C++, not forget to call the command which sets default options
 (``finufft_default_opts``)
 before you start changing them or passing them to FINUFFT.


Summary and quick advice
-------------------------------------

Here is a 1-line summary of each option, taken from the code
(the header ``include/nufft_opts.h``):

.. literalinclude:: ../include/nufft_opts.h
   :start-after: @opts_start
   :end-before: @opts_end

Here are their default settings (from ``src/finufft.cpp:finufft_default_opts``):

.. literalinclude:: ../src/finufft.cpp
   :start-after: @defopts_start
   :end-before: @defopts_end
  
As for quick advice, the main options you'll want to play with are:
  
- ``modeord`` to flip the Fourier mode ordering
- ``debug`` to look at timing output (to determine if your problem is spread/interpolation dominated, vs FFT dominated)
- ``nthreads`` to run with a different number of threads than the current maximum available through OpenMP (a large number can sometimes be detrimental, and very small problems can sometimes run faster on 1 thread)
- ``fftw`` to try slower plan modes which give faster transforms. The next natural one to try is ``FFTW_MEASURE`` (look at the FFTW3 docs)

See :ref:`Troubleshooting <trouble>` for good advice on trying options, and read the full options descriptions below.

  .. warning::
Some of the options are experts-only, and will result in slow or incorrect results. Please test them in a small known test case so you understand the effect.


Documentation of all options
-----------------------------

Data handling options
~~~~~~~~~~~~~~~~~~~~~

``modeord``: Fourier coefficient frequency index ordering in each dimension. For type 1, this is for the output; for type 2 the input. It has no effect in type 3.
For example, if ``N1=8`` in a 1D type 1 or type 2 transform:

* if ``opts.modeord=0``: frequency indices are ordered ``-4,-3,-2,-1,0,1,2,3`` (CMCL ordering)

* if ``opts.modeord=1``: frequency indices are ordered ``0,1,2,3,-4,-3,-2,-1`` (FFT ordering)

``chkbnds``: whether to check the nonuniform points lie in the correct bounds.

*  ``opts.chkbnd=0``: input nonuniform points in the arrays `x`, `y`, `z`, are fed straight into the spreader which assumes (for speed) that they lie in :math:`[-3\pi,3\pi)`. Points outside of this will then cause a segfault.

*  ``opts.chkbnd=1``: the nonuniform points are checked to lie in this interval, and if any are found not to, the library exits with an error code and message to stderr. The trade-off is that simply doing this checking can lose several % in overall speed, especially in low-precision 3D transforms.
  

Diagnostic options
~~~~~~~~~~~~~~~~~~~~~~~

``debug``: Controls the amount of debug/timing output to stdout.
0 is silent, 1 prints some information, and 2 more.

``spread_debug``: Controls the amount of debug/timing output from the spreader/interpolator. 0 is silent, 1 prints some timing information, and 2 can print thousands of lines since it includes one line per `subproblem'.

``showwarn``: Whether to print warnings (which go to stderr).
0 suppresses such warnings, while 1 prints them.


Algorithm performance options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``nthreads``: Number of threads to use. This sets the number of threads FINUFFT will use in FFTW, bin-sorting, and spreading/interpolation steps. This number of threads also controls the batch size for vectorized transforms (ie ``ntr>1`` :ref:`here <c>`). Setting ``opts.nthreads=0`` uses all threads available (up to an internal maximum that has been chosen based on performance; see ``MAX_USEFUL_NTHREADS`` in ``include/defs.h``). For repeated small problems it can be advantageous to use a small number, such as 1.

``fftw``: FFTW planner flags. This number is simply passed to FFTW's planner;
the flags are documented `here <http://www.fftw.org/fftw3_doc/Planner-Flags.html#Planner-Flags>`_.
A good first choice is ``FFTW_ESTIMATE``; however if you will be making multiple calls, consider ``FFTW_MEASURE``, which could spend many seconds planning, but will give a faster run-time when called again from the same process. These macros are bit-wise flags defined in ``/usr/include/fftw3.h`` on a linux system; they currently have the values ``FFTW_ESTIMATE=64`` and ``FFTW_MEASURE=0``. Note that FFTW plans are saved (by FFTW's library)
automatically from call to call in the same executable (incidentally, also in the same MATLAB/octave or python session); there is a small overhead for lookup of such plans, which with many repeated small problems can motivate use of the :ref:`guru interface <guru>`.

``spread_sort``: Sorting mode within the spreader/interpolator. 0 never sorts, 1 always sorts, and 2 uses a heuristic to decide whether to sort or not. Generally it is not worth sorting in 1D type 2 transforms, or when the number of nonuniform points is small.


***



``upsampfac``: This is the internal factor by which the FFT is larger than
the number of requested modes in each dimension. We have built efficient kernels
for only two settings: ``upsampfac=2.0`` (standard), and ``upsampfac=1.25``
(lower RAM, smaller FFTs, but wider spreading kernel).
The latter can be much faster when the number of nonuniform points is similar or
smaller to the number of modes, and/or if low accuracy is required.
It is especially much faster for type 3 transforms.
However, the kernel widths :math:`w` are about 50% larger in each dimension,
which can lead to slower spreading (it can also be faster due to the smaller
size of the fine grid).
Thus only 9-digit accuracy can currently be reached when using
``upsampfac=1.25``.

The remaining options only are relevant for multiple-vector calls, that is,
using the simple interfaces containing the word "many", or the guru interface with ``ntrans`` > 1:

``spread_thread``: control how multithreading is used to spread/interpolate each batch of data.

- 0: makes an automatic choice.
  
- 1: acts on each vector in the batch in sequence, using multithreaded spread/interpolate. It can be slightly better than 2 for large problems.

- 2: acts on all vectors in batch simultaneously, assigning each a thread which performs single-threaded spread/interpolate. (This was used by Melody Shih for the original "2dmany" interface in 2018.) It is much better than 1 for all but large problems.

- 3: like 2 except allowing nested OMP parallelism, so multi-threaded spread-interpolate is used. (This was used by Andrea Malleo in 2019.) I have not yet found a case where this beats both 1 and 2.
  
``maxbatchsize``: set the largest batch size of data vectors. 0 makes an automatic choice. If you are unhappy with this, then for small problems it should equal the number of threads, while for large problems it appears that 1 is better
(since otherwise too much simultaneous RAM movement occurs).



*** REWRITE AND SPLIT UP:


Usage and design notes
**********************

- We strongly recommend you use ``upsampfac=1.25`` for type-3; it
  reduces its run-time from around 8 times the types 1 or 2, to around 3-4
  times. It is often also faster for type-1 and type-2, at low precisions.

- Sizes >=2^31 have been tested for C++ drivers (``test/finufft?d_test.cpp``), and
  work fine, if you have enough RAM.
  In fortran the interface is still 32-bit integers, limiting to
  array sizes <2^31. The fortran interface needs to be improved.

- C++ is used for all main libraries, almost entirely avoiding object-oriented code. C++ ``std::complex<double>`` (macroed to ``CPX`` and sometimes ``dcomplex``) and FFTW complex types are mixed within the library, since to some extent our library is a glorified driver for FFTW. FFTW was considered universal and essential enough to be a dependency for the whole package.

- There is a hard-defined limit of ``1e11`` for the size of internal FFT arrays, set in ``defs.h`` as ``MAX_NF``: if your machine has RAM of order 1TB, and you need it, set this larger and recompile. The point of this is to catch ridiculous-sized mallocs and exit gracefully. Note that mallocs smaller than this, but which still exceed available RAM, cause segfaults as usual. For simplicity of code, we do not do error checking on every malloc.

- As a spreading kernel function, we use a new faster simplification of the Kaiser--Bessel kernel, and eventually settled on piecewise polynomial approximation of this kernel.  At high requested precisions, like the Kaiser--Bessel, this achieves roughly half the kernel width achievable by a truncated Gaussian. Our kernel is exp(-beta.sqrt(1-(2x/W)^2)), where W = nspread is the full kernel width in grid units. This (and Kaiser--Bessel) are good approximations to the prolate spheroidal wavefunction of order zero (PSWF), being the functions of given support [-W/2,W/2] whose Fourier transform has minimal L2 norm outside of a symmetric interval. The PSWF frequency parameter (see [ORZ]) is c = pi.(1-1/2sigma).W where sigma is the upsampling parameter. See our paper in the references.
