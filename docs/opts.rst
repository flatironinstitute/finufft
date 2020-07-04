.. _opts:

Options parameters
==================

Aside from the mandatory inputs (dimension, type, sign of the imaginary
unit, tolerance, nonuniform points, strengths or coefficients),
FINUFFT has optional parameters.
These adjust the workings of the algorithm, but
still result in the same numerical output, to within tolerance.
The exception is ``modeord`` which changes the Fourier mode ordering
in types 1 and 2.
Sensible default options are chosen, so that the new user need not worry about
changing them.
However, users wanting to try to increase speed or see more debugging output
will want to change options from their defaults.
This is usually done by defining a options structure
(under the hood this is a simple C struct),
first setting default values, then changing whichever need to be changed,
then passing the structure to either the simple, vectorized, or guru makeplan
FINUFFT routines, as here from C++::

.. code-block:: C++
                
  // (... set up M,x,c,tol,N, and allocate F here...)
  nufft_opts opts;
  finufft_default_opts(&opts);
  opts.upsampfac = 1.25;
  int ier = finufft1d1(M,x,c,+1,tol,N,F,&opts);

This example sets a non-default upsampling factor :math:`\sigma`.

  .. warning::
In C/C++, not forget to call the command which sets default options
(``finufft_default_opts``)
before you start changing them or passing them to FINUFFT.

Summary of options and quick advice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is the 1-line summary of each option, with the full specifications below
(see the header ``include/nufft_opts.h``)::

  int debug;          // 0: silent, 1: text basic timing output
  int spread_debug;   // passed to spread_opts, 0 (no text) 1 (some) or 2 (lots)
  int spread_sort;    // passed to spread_opts, 0 (don't sort) 1 (do) or 2 (heuristic)
  int spread_kerevalmeth; // "     spread_opts, 0: exp(sqrt()), 1: Horner ppval (faster)
  int spread_kerpad;  // passed to spread_opts, 0: don't pad to mult of 4, 1: do
  int chkbnds;        // 0: don't check if input NU pts in [-3pi,3pi], 1: do
  int fftw;           // 0:FFTW_ESTIMATE, or 1:FFTW_MEASURE (slow plan, faster run)
  int modeord;        // 0: CMCL-style increasing mode ordering (neg to pos), or
                      // 1: FFT-style mode ordering (affects type-1,2 only)
  double upsampfac;   // upsampling ratio sigma, either 2.0 (standard) or 1.25 (small FFT)
  int spread_thread;  // for ntrans>1 only. 0: auto, 1: sequential multithreaded, 2: parallel singlethreaded, 3: nested multithreaded
  int maxbatchsize;   // for ntrans>1 only. max chunk size of data vectors. 0: auto

Here are their default settings (set in ``src/finufft.cpp:finufft_default_opts``)::

  debug = 0;
  spread_debug = 0;
  spread_sort = 2;
  spread_kerevalmeth = 1;
  spread_kerpad = 1;
  chkbnds = 0;
  fftw = FFTW_ESTIMATE;
  modeord = 0;
  upsampfac = 2.0;
  spread_thread = 0;
  maxbatchsize = 0;
  
To get the fastest run-time, we recommend that you experiment firstly with:
``fftw``, ``upsampfac``, and ``spread_sort``, detailed below.
If you are having crashes, set ``chkbnds=1`` to see if illegal ``x`` non-uniform point coordinates are being input.

Notes on various options:

``spread_sort``: the default setting is ``spread_sort=2``
which applies the following heuristic rule: in 2D or 3D always sort, but in 1D,
only sort if N (number of modes) > M/10 (where M is number of nonuniform pts).

``fftw``:
The default FFTW plan is ``FFTW_ESTIMATE``; however if you will be making multiple calls, consider ``fftw=FFTW_MEASURE``, which could spend many seconds planning, but will give a faster run-time when called again. Note that FFTW plans are saved (by FFTW's library)
automatically from call to call in the same executable (incidentally, also in the same MATLAB/octave or python session).

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
