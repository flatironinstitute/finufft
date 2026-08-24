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
  finufft_opts opts;
  finufft_default_opts(&opts);
  opts.debug = 1;
  int ier = finufft1d1(M,x,c,+1,tol,N,F,&opts);

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

Here are their default settings (from ``include/finufft/plan.hpp:finufft_default_opts_t``):

.. literalinclude:: ../include/finufft/plan.hpp
   :start-after: @defopts_start
   :end-before: @defopts_end

As for quick advice, the main options you'll want to play with are:

- ``upsampfac`` to trade-off between spread/interpolate vs FFT speed and RAM
- ``modeord`` to flip ("fftshift") the Fourier mode ordering
- ``debug`` to look at timing output (to determine if your problem is spread/interpolation dominated, vs FFT dominated)
- ``nthreads`` to run with a different number of threads than the current maximum available through OpenMP (a large number can sometimes be detrimental, and very small problems can sometimes run faster on 1 thread)
- ``fftw`` to try slower FFTW plan modes which give faster transforms. The next natural one to try is ``FFTW_MEASURE`` (look at the FFTW3 docs)

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

.. _sionly:

**spreadinterponly**: [only has effect for type 1 or 2.]
Controls whether a NUFFT is performed, or only spreading or interpolation.
For experts only.

* If ``0`` do the NUFFT as intended.

* If ``1``, omit the FFT and deconvolution
  (diagonal division by kernel Fourier transform) steps, thus returning
  *garbage answers as a NUFFT*, but allowing experts to perform solely
  spreading (if type 1) or solely interpolation (if type 2) via
  the FINUFFT API.  The spreading is onto the grid of the
  user-given size (``N1`` in x, ``N2`` in y, etc), with grid points
  located at coordinates $\{-\pi, -\pi+h, \dots, \pi-h\}$ in each
  dimension, where $h = 2\pi/N$ is the spacing for that dimension ($N$
  here meaning ``N1``, etc). Interpolation is from that same grid.  The
  kernel (width and shape parameter) is determined by ``tol`` and
  ``opts.upsampfac``, just as it would be in an actual NUFFT. Note that
  the upsampling factor here only controls the kernel; the grid size
  never differs from ``N1``, etc.  The kernel is not directly
  accessible, leaving the user to figure out how to make use of this
  interface to extract the actual kernel function.  This provides a
  convenient interface to our ``spreadinterp`` module
  (including looping over multiple vectors, if ``ntransf>1``).

  .. note:: The known use-case of ``spreadinterponly=1`` is estimating so-called density compensation weights, conventionally used in MRI (see `MRI-NUFFT <https://mind-inria.github.io/mri-nufft/nufft.html>`_). It may also be useful in spectral Ewald or other scientific applications.



Diagnostic options
~~~~~~~~~~~~~~~~~~~~~~~

**debug**: Controls the amount of overall debug/timing output to stdout.

* ``debug=0`` : silent

* ``debug=1`` : prints some information

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

As of v2.6.0 both directions sort. The heuristic sorts unless a single thread already holds the whole fine grid in its L2 cache. Everywhere else a sort costs one pass over the points and buys every later pass a fine grid that stays in cache. The sort also groups the points into the cache tiles that spreading and interpolation take as subproblems, so ``spread_sort=0`` leaves both directions with one chunk of the point list per thread instead.
Feel free to try experimenting here; if you have highly-structured nonuniform point ordering (such as coming from polar-grid or propeller-type MRI k-points) it may be advantageous not to sort.

**upsampfac**: This is the internal factor $\sigma$ by which the FFT (fine grid)
is chosen larger than
the number of requested modes in each dimension, for type 1 and 2 transforms. For type 3 transforms this factor gets squared, due to type 2 nested in a type-1-spreading operation, so has even more influence.
As of v2.5.0, due to on-the-fly polynomial coefficient fitting, the kernel is equally efficient for an arbitrary upsampling factor greater than 1, but the useful range is around 1.2 up to 3.0.

* ``upsampfac=0.0`` : use heuristics to choose a good ``upsampfac`` based on the problem.
 The value chosen is visible in the text output via setting ``debug>=1``. This default setting is recommended for most users; however, if you seek more performance you may want to set if 

* ``upsampfac>1.0`` : fix the upsampling factor, overriding the heuristic choice. A standard setting is 2 (which is good for achieving 9-digit or more accuracy), while a typical "low" setting is 1.25 (this reduces the RAM and FFT costs, and is good for up to 5-digit accuracy, unless the density M/N is high enough that its 50% wider spreading kernel would be counterproductive). Low upsampfac is especially efficient for type 3 transforms. Because the kernel width is limited to 16, only 9-digit accuracy can be reached when using ``upsampfac=1.25``, for instance.

**spread_thread**: DEPRECATED as of v2.6.0, and ignored (the field is retained for ABI compatibility, and setting it emits a compiler deprecation warning in C++). Both directions now use all threads on the whole batch, so there is nothing left to choose. Both directions fold the batch loop into the loop over subproblems (the load-balanced scheme of Sec. 5.2 of our paper [FIN] in the :doc:`references <refs>`), so (vector, subproblem) pairs are what get assigned to threads. Only spreading writes the fine grid, so the paper's ``omp critical`` on the add back becomes a per-vector lock, or atomic writes above ``spread_nthr_atomic`` threads on one grid; interpolation reads the grid and takes no lock at all.

Setting it to anything other than its ``0`` default prints a runtime warning (suppressed by ``showwarn=0``), so callers through the Fortran, Python and MATLAB wrappers - which cannot see the C++ attribute - are told it is ignored.

.. note::

  Historical note: this selected between multithreaded spread/interpolate on each vector of the batch in sequence (``1``), and one thread per vector with all vectors at once (``2``, used by Melody Shih for the original "2dmany" interface in 2018); a further option ``3`` allowing nested OMP parallelism (Andrea Malleo, 2019) was already removed. ``2`` was the automatic choice for ``ntr>1``, but only ever kept as many threads busy as the batch was long: OMP nesting is off by default, so each vector still split into ``nthreads`` subproblems, which one thread then ran in sequence.


**maxbatchsize**:  in the case of multiple transforms per call (``ntr>1``, or the "many" interfaces), set the largest batch size of data vectors.
Here ``0`` makes an automatic choice. If you are unhappy with this, then for small problems it should equal the number of threads, while for large problems it appears that ``1`` often better (since otherwise too much simultaneous RAM movement occurs). Some further work is needed to optimize this parameter.

**spread_nthr_atomic**: if non-negative: for numbers of threads up to this value, one lock per fine grid guards ``add_wrapped_subgrid`` in spreading (type 1 transforms). Above this value, instead OMP atomic writes are used, which scale better for large thread numbers. If negative, the heuristic default in the spreader is used, set in ``FINUFFT_PLAN_T::setup_spreadinterp()`` in ``include/finufft/makeplan.hpp``.

**spread_max_sp_size**: DEPRECATED as of v2.6.0, and ignored (the field is retained for ABI compatibility, and setting it emits a compiler deprecation warning in C++). It overrode the maximum subproblem size for multithreaded spreading. A subproblem is now a cache tile, so its size follows from the cache, and an unsorted point list is one tile cut one subproblem per thread.

**spread_kerformula**: ``0`` uses the default spreading (gridding) kernel with default shape choice; ``7``, ``8`` and ``9`` select among shape parameter choices for it. As of v2.6.0 the prolate spheroidal wavefunction (PSWF) is the only kernel: the legacy ES, Kaiser--Bessel and cosh-type formulas (``1``--``6``, available up to v2.5.0) have been removed and now return an error. Only developers should mess with this parameter; users should leave it at default.

**spread_kerevalmeth**: [DEPRECATED] Kernel evaluation method in spreader/interpolator; retained only for API compatibility and documentation. The library now always uses the Horner piecewise-polynomial evaluation internally (the historical ``=1`` choice). Setting this field has no effect.

**spread_kerpad**: [DEPRECATED] This option historically controlled padding to help SIMD vectorization for the removed direct-evaluation method. It is ignored by the library.

Like ``spread_thread``, each of the deprecated options above prints a runtime warning when set away from its default, in addition to the C++ compiler deprecation warning.



Thread safety options (advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With DUCC0 as the FFT, there are no thread safety issues.
However, with FFTW as the FFT library, FINUFFT is thread safe so long as no other threads are calling FFTW plan creation/destruction routines independently of FINUFFT. If these FFTW routines are called outside of FINUFFT, then the program is liable to crash. In most cases, the calling program can simply call the FFTW routine ``fftw_make_planner_thread_safe()`` before threading out and thread safety will be maintained. However, in instances where this is less desirable, we provide a means to provide your own FFTW locking mechanism. The following example code should exercise FFTW thread safety, and can be built with ``c++ thread_test.cpp -o thread_test -lfinufft -lfftw3_threads -lfftw3 -fopenmp -std=c++11``, assuming the finufft include and library paths are set.

.. code-block:: C++


  // thread_test.cpp
  #include <vector>
  #include <mutex>
  #include <complex>

  #include <fftw3.h>
  #include <finufft.h>
  #include <omp.h>

  constexpr int N = 65384;

  void locker(void *lck) { reinterpret_cast<std::recursive_mutex *>(lck)->lock(); }
  void unlocker(void *lck) { reinterpret_cast<std::recursive_mutex *>(lck)->unlock(); }

  int main() {
    int64_t Ns[3]; // guru describes mode array by vector [N1,N2..]
    Ns[0] = N;
    std::recursive_mutex lck;

    finufft_opts opts;
    finufft_default_opts(&opts);
    opts.nthreads = 1;
    opts.debug = 0;
    opts.fftw_lock_fun = locker;
    opts.fftw_unlock_fun = unlocker;
    opts.fftw_lock_data = reinterpret_cast<void *>(&lck);

    // random nonuniform points (x) and complex strengths (c)
    std::vector<std::complex<double>> c(N);

    // init FFTW threads
    fftw_init_threads();

    // FFTW and FINUFFT execution using OpenMP parallelization
    #pragma omp parallel for
    for (int j = 0; j < 100; ++j) {
      // allocate output array for FFTW...
      std::vector<std::complex<double>> F1(N);

      // FFTW plan
      lck.lock();
      fftw_plan_with_nthreads(1);
      fftw_plan plan = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(c.data()),
                                        reinterpret_cast<fftw_complex*>(F1.data()),
                                        FFTW_FORWARD, FFTW_ESTIMATE);
      fftw_destroy_plan(plan);
      lck.unlock();

      // FINUFFT plan
      finufft_plan nufftplan;
      finufft_makeplan(1, 1, Ns, 1, 1, 1e-6, &nufftplan, &opts);
      finufft_destroy(nufftplan);
    }

    return 0;
  }

**fftw_lock_fun**:  ``void (fun*)(void *)`` C-style callback function to lock calls to FFTW plan manipulation routines. A ``nullptr`` or ``0`` value will be ignored. If non-null, ``fftw_unlock_fun`` must also be set.

**fftw_unlock_fun**: ``void (fun*)(void *)`` C-style callback function to unlock calls to FFTW plan manipulation routines. A ``nullptr`` or ``0`` value will be ignored. If non-null, ``fftw_lock_fun`` must also be set.

**fftw_lock_data**:  ``void *data`` pointer, typically to the lock object itself. Pointer will be passed to ``fftw_lock_fun`` and ``fftw_unlock_fun`` if they are set.
