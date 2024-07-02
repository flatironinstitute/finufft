.. _opts:

Options parameters (CPU)
========================

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
  finufft_opts* opts;
  finufft_default_opts(opts);
  opts->debug = 1;
  int ier = finufft1d1(M,x,c,+1,tol,N,F,opts);

This setting produces more timing output to ``stdout``.

.. warning::
   
 In C/C++ and Fortran, don't forget to call the command which sets default options
 (``finufft_default_opts`` or ``finufftf_default_opts``)
 before you start changing them and passing them to FINUFFT.


Summary and quick advice
-------------------------------------

Here is a 1-line summary of each option, taken from the code
(the header ``include/finufft_opts.h``):

.. literalinclude:: ../include/finufft_opts.h
   :start-after: @opts_start
   :end-before: @opts_end

Here are their default settings (from ``src/finufft.cpp:finufft_default_opts``):

.. literalinclude:: ../src/finufft.cpp
   :start-after: @defopts_start
   :end-before: @defopts_end
  
As for quick advice, the main options you'll want to play with are:
  
- ``modeord`` to flip ("fftshift") the Fourier mode ordering
- ``debug`` to look at timing output (to determine if your problem is spread/interpolation dominated, vs FFT dominated)
- ``nthreads`` to run with a different number of threads than the current maximum available through OpenMP (a large number can sometimes be detrimental, and very small problems can sometimes run faster on 1 thread)
- ``fftw`` to try slower plan modes which give faster transforms. The next natural one to try is ``FFTW_MEASURE`` (look at the FFTW3 docs)

See :ref:`Troubleshooting <trouble>` for good advice on trying options, and read the full options descriptions below.

.. warning::
  Some of the options are for experts only, and will result in slow or incorrect results. Please test options in a small known test case so that you understand their effect.


Documentation of all options
-----------------------------

Data handling options
~~~~~~~~~~~~~~~~~~~~~

.. _modeord:

**modeord**: Fourier coefficient frequency index ordering in every dimension. For type 1, this is for the output; for type 2 the input. It has no effect in type 3. Here we use ``N`` to denote the size in any of the relevant dimensions:

* if ``modeord=0``: frequency indices are in increasing ordering,
  namely $\{-N/2,-N/2+1,\dots,N/2-1\}$ if $N$ is even, or
  $\{-(N-1)/2,\dots,(N-1)/2\}$ if $N$ is odd.
  For example, if ``N=6`` the indices are ``-3,-2,-1,0,1,2``,
  whereas if ``N=7`` they are ``-3,-2,-1,0,1,2,3``.
  This is called "CMCL ordering" since it is that of the CMCL NUFFT.

* if ``modeord=1``: frequency indices are ordered as in the usual FFT,
  increasing from zero then jumping to negative indices half way along,
  namely $\{0,1,\dots,N/2-1,-N/2,-N/2+1,\dots,-1\}$ if $N$ is even, or
  $\{0,1,\dots,(N-1)/2,-(N-1)/2,\dots,-1\}$ if $N$ is odd.
  For example, if ``N=6`` the indices are ``0,1,2,-3,-2,-1``,
  whereas if ``N=7`` they are ``0,1,2,3,-3,-2,-1``.

  .. note:: The index *sets* are the same in the two ``modeord`` choices; their ordering differs only by a cyclic shift. The FFT ordering cyclically shifts the CMCL indices $\mbox{floor}(N/2)$ to the left (often called an "fftshift").

**chkbnds**: [DEPRECATED] has no effect.
  

Diagnostic options
~~~~~~~~~~~~~~~~~~~~~~~

**debug**: Controls the amount of overall debug/timing output to stdout.

* ``debug=0`` : silent
  
* ``debug=1`` : print some information

* ``debug=2`` : prints more information

**spread_debug**: Controls the amount of debug/timing output from the spreader/interpolator.

* ``spread_debug=0`` : silent

  * ``spread_debug=1`` : prints some timing information

  * ``spread_debug=2`` : prints lots. This can print thousands of lines since it includes one line per *subproblem*.

   
**showwarn**: Whether to print warnings (these go to stderr).
    
* ``showwarn=0`` : suppresses such warnings
  
* ``showwarn=1`` : prints warnings


Algorithm performance options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**nthreads**: (Ignored in single-threaded library builds.) If positive, sets the number of threads to use throughout (multi-threaded build of) library, or if ``0`` uses the maximum number of threads available according to OpenMP. In the positive case, no cap is placed on this number. This number of threads is passed to bin-sorting (which may choose to use less threads), but is adhered to in FFTW and spreading/interpolation steps. This number of threads (or 1 for single-threaded builds) also controls the batch size for vectorized transforms (ie ``ntr>1`` :ref:`here <c>`).
For medium-to-large transforms, ``0`` is usually recommended.
However, for (repeated) small transforms it can be advantageous to use a small number, even as small as ``1``.

**fftw**: FFTW planner flags. This number is simply passed to FFTW's planner;
the flags are documented `here <http://www.fftw.org/fftw3_doc/Planner-Flags.html#Planner-Flags>`_.
A good first choice is ``FFTW_ESTIMATE``; however if you will be making multiple calls, consider ``FFTW_MEASURE``, which could spend many seconds planning, but will give a faster run-time when called again from the same process. These macros are bit-wise flags defined in ``/usr/include/fftw3.h`` on a linux system; they currently have the values ``FFTW_ESTIMATE=64`` and ``FFTW_MEASURE=0``. Note that FFTW plans are saved (by FFTW's library)
automatically from call to call in the same executable (incidentally, also in the same MATLAB/octave or python session); there is a small overhead for lookup of such plans, which with many repeated small problems can motivate use of the :ref:`guru interface <guru>`.

**spread_sort**: Sorting mode within the spreader/interpolator.

* ``spread_sort=0`` : never sorts
* ``spread_sort=1`` : always sorts
* ``spread_sort=2`` : uses a heuristic to decide whether to sort or not.

The heuristic bakes in empirical findings such as: generally it is not worth sorting in 1D type 2 transforms, or when the number of nonuniform points is small.
Feel free to try experimenting here; if you have highly-structured nonuniform point ordering (such as coming from polar-grid or propeller-type MRI k-points) it may be advantageous not to sort.

**spread_kerevalmeth**: Kernel evaluation method in spreader/interpolator.
This should not be changed from its default value, unless you are an
expert wanting to compare against outdated

* ``spread_kerevalmeth=0`` : direct evaluation of ``sqrt(exp(beta(1-x*x)))`` in the ES kernel. This is outdated, and it's only possible use could be in exploring upsampling factors :math:`\sigma` different from standard (see below).

* ``spread_kerevalmeth=1`` : use Horner's rule applied to piecewise polynomials with precomputed coefficients. This is faster, less brittle to compiler/glibc/CPU variations, and is the recommended approach. It only works for the standard upsampling factors listed below.

**spread_kerpad**: whether to pad the number of direct kernel evaluations per dimension and per nonuniform point to a multiple of four; this can help SIMD vectorization. It only applies to the (outdated) ``spread_kerevalmeth=0`` choice.
There is thus little reason for the nonexpert to mess with this option.

* ``spread_kerpad=0`` : do not pad

* ``spread_kerpad=1`` : pad to next multiple of four


**upsampfac**: This is the internal real factor by which the FFT (fine grid)
is chosen larger than
the number of requested modes in each dimension, for type 1 and 2 transforms.
We have built efficient kernels
for only two settings, as follows. Otherwise, setting it to zero chooses a good heuristic:

* ``upsampfac=0.0`` : use heuristics to choose ``upsampfac`` as one of the below values, and use this value internally. The value chosen is visible in the text output via setting ``debug>=2``. This setting is recommended for basic users; however, if you seek more performance it is quick to try the other of the below.

* ``upsampfac=2.0`` : standard setting of upsampling. This is necessary if you need to exceed 9 digits of accuracy.

* ``upsampfac=1.25`` : low-upsampling option, with lower RAM, smaller FFTs, but wider spreading kernel. The latter can be much faster than the standard when the number of nonuniform points is similar or smaller to the number of modes, and/or if low accuracy is required. It is especially much (2 to 3 times) faster for type 3 transforms. However, the kernel widths :math:`w` are about 50% larger in each dimension, which can lead to slower spreading (it can also be faster due to the smaller size of the fine grid). Because the kernel width is limited to 16, currently, thus only 9-digit accuracy can currently be reached when using ``upsampfac=1.25``.

**spread_thread**: in the case of multiple transforms per call (``ntr>1``, or the "many" interfaces), controls how multithreading is used to spread/interpolate each batch of data.

* ``spread_thread=0`` : makes an automatic choice between the below. Recommended.
  
* ``spread_thread=1`` : acts on each vector in the batch in sequence, using multithreaded spread/interpolate on that vector. It can be slightly better than ``2`` for large problems.
    
* ``spread_thread=2`` : acts on all vectors in a batch (of size chosen typically to be the number of threads) simultaneously, assigning each a thread which performs a single-threaded spread/interpolate.  It is much better than ``1`` for all but large problems. (Historical note: this was used by Melody Shih for the original "2dmany" interface in 2018.)

  .. note::
  
    Historical note: A former option ``3`` has been removed. This was like ``2`` except allowing nested OMP parallelism, so multi-threaded spread-interpolate was used for each of the vectors in a batch in parallel. This was used by Andrea Malleo in 2019. We have not yet found a case where this beats both ``1`` and ``2``, hence removed it due to complications with changing the OMP nesting state in both old and new OMP versions.

     
**maxbatchsize**:  in the case of multiple transforms per call (``ntr>1``, or the "many" interfaces), set the largest batch size of data vectors.
Here ``0`` makes an automatic choice. If you are unhappy with this, then for small problems it should equal the number of threads, while for large problems it appears that ``1`` often better (since otherwise too much simultaneous RAM movement occurs). Some further work is needed to optimize this parameter.

**spread_nthr_atomic**: if non-negative: for numbers of threads up to this value, an OMP critical block for ``add_wrapped_subgrid`` is used in spreading (type 1 transforms). Above this value, instead OMP atomic writes are used, which scale better for large thread numbers. If negative, the heuristic default in the spreader is used, set in ``src/spreadinterp.cpp:setup_spreader()``.

**spread_max_sp_size**: if positive, overrides the maximum subproblem (chunking) size for multithreaded spreading (type 1 transforms). Otherwise the default in the spreader is used, set in ``src/spreadinterp.cpp:setup_spreader()``, which we believe is a decent heuristic for Intel i7 and xeon machines.
