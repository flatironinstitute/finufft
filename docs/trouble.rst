.. _trouble:

Troubleshooting
===============

If you are having issues (segfaults, slowness, "wrong" answers, etc),
there is a high probability it is something we already know about, so
please first read all of the advice below in the section relevant
to your problem: math, speed, or crashing.


Mathematical "issues" and advice
********************************

- When requested tolerance is around $10^{-14}$ or less in double-precision,
  or $10^{-6}$ or less in single-precision, it
  will most likely be impossible for FINUFFT (or any other NUFFT library)
  to achieve this, due to inevitable round-off error.
  Here, "error" is to be understood relative to the norm of the returned vector
  of values.
  This is especially true when there is a large number of modes in
  any single dimension ($N_1$, $N_2$ or $N_3$), since this empirically
  scales the round-off error (fortunately, round-off does not appear to scale
  with the total $N$ or $M$).
  Such round-off error is analysed and measured in Section 4.2 of our `SISC paper <https://arxiv.org/abs/1808.06736>`_.

- If you request a tolerance that FINUFFT knows it cannot achieve, it will return ``ier=1`` after performing transforms as accurately as it can. However, the status ``ier=0`` does not imply that the requested accuracy *was* achieved, merely that parameters were chosen to give this estimated accuracy, if possible. As our SISC paper shows, for typical situations, relative $\ell_2$ errors match the requested tolerances over a wide range.
  Users should always check *convergence* (by, for instance, varying ``tol`` and measuring any changes in their results); this is generally true in scientific computing.

- On the above topic, strangely, in single-precision, requesting tolerance
  of $10^{-7}$ or $10^{-6}$ can give slightly *worse* accuracy than $10^{-5}$ or $10^{-4}$. We are looking into this. It is usually best to request at least 2--3 digits above the respective machine precision, for either single or double precision.
  
- The type 1 and type 2 transforms are adjoints but **not inverses of each other** (unlike in the plain FFT case, where, up to a constant factor $N$, the adjoint is the inverse). Therefore, if you are not getting the expected answers, please check that you have not made this assumption. In the :ref:`tutorials <tut>` we will add examples showing how to invert the NUFFT; also see `NFFT3 inverse transforms <https://www-user.tu-chemnitz.de/~potts/nfft/infft.php>`_.


Speed issues and advice
***********************

If FINUFFT is slow (eg, less than $10^6$ nonuniform points per second), here is some advice:

- Try printing debug output to see step-by-step progress by FINUFFT. Do this by setting ``opts.debug`` to 1 or 2 then looking at the timing information.

- Try reducing the number of threads, either those available via OpenMP, or via ``opts.nthreads``, perhaps down to 1 thread, to make sure you are not having collisions between threads, or slowdown due to thread overheads. Hyperthreading (more threads than physical cores) rarely helps much. Thread collisions are possible if large problems are run with a large number of (say more than 64) threads. Another case causing slowness is very many repetitions of small problems; see ``test/manysmallprobs`` which exceeds $10^7$ points/sec with one thread via the guru interface, but can get ridiculously slower with many threads; see https://github.com/flatironinstitute/finufft/issues/86

- Try setting a crude tolerance, eg ``tol=1e-3``. How many digits do you actually need? This has a big effect in higher dimensions, since the number of flops scales like $(\log 1/\epsilon)^d$, but not quite as big an effect as this scaling would suggest, because in higher dimensions the flops/RAM ratio is higher.

- If type 3, make sure your choice of points does not have a massive *space-bandwidth product* (ie, product of the volumes of the smallest $d$-dimension axes-aligned cuboids enclosing the nonuniform source and the target points); see Remark 5 of our `SISC paper <https://arxiv.org/abs/1808.06736>`_.
  In short, if the spreads of $\mathbf{x}_j$ and of $\mathbf{s}_k$ are both big, you may be in trouble.
  This can lead to enormous fine grids and hence slow FFTs. Set ``opts.debug=1`` to examine the ``nf1``, etc, fine grid sizes being chosen, and the array allocation sizes. If they are huge, consider direct summation, as discussed :ref:`here <need>`.
  
- The timing of the first FFTW call is complicated, depending on the FFTW flags (plan mode) used. This is really an
  `FFTW planner flag usage <http://www.fftw.org/fftw3_doc/Planner-Flags.html#Planner-Flags>`_ question.
  Such issues are known, and modes benchmarked in other documentation, eg for 2D in `poppy <https://poppy-optics.readthedocs.io/en/stable/fft_optimization.html>`_. In short, using more expensive FFTW planning modes like ``FFTW_MEASURE`` can give better performance for repeated FFTW calls, but be **much** more expensive in the first (planning) call. This is why we choose ``FFTW_ESTIMATE`` as our default ``opts.fftw`` option.

- Check that you are not using too much RAM, hence swapping to hard disk or SSD. The multithreaded type-1 spreader can use up to another fine grid's worth of storage in the form of subgrids. If RAM is too large, try overriding ``opts.spread_max_sp_size`` to a nonzero value smaller than the default value set in ``src/spreadinterp.cpp:setup_spreader()``, to reduce RAM. However, note that this may slow it down, because we have built in a decent heuristic for the default.
    
- Make sure you did not override ``opts.spread_sort``, which if set to zero
  does no sorting, which can give very slow RAM access if the nonuniform points
  are ordered poorly (eg randomly) in larger 2D or 3D problems.

- Are you calling the simple interface a huge number of times for small problems, but these tasks have something in common (number of modes, or locations of nonuniform points)? If so, try the "many vector" or guru interface, which removes overheads in repeated FFTW plan look-up, and in bin-sorting. They can be 10-100x faster.


Crash (segfault) issues and advice
****************************************

- The most common problem is passing in pointers to the wrong size of object, eg, single vs double precision, or int32 vs int64. The library includes both precisions, so make sure you are calling the correct one (commands begin ``finufft`` for double, ``finufftf`` for single).

- If you use C++/C/Fortran and tried to change options, did you forget to call ``finufft_default_opts`` first?

- Maybe you have switched off nonuniform point bounds checking (``opts.chkbnds=0``) for a little extra speed? Try switching it on again to catch illegal coordinates.

- Thread-safety: are you calling FINUFFT from inside a multithreaded block of code without setting ``opts.nthreads=1``? If ``gdb`` indicates crashes during FFTW calls, this is another sign.
  
- To isolate where a crash is occurring, set ``opts.debug`` to 1 or 2, and check the text output of the various stages. With a debug setting of 2 or above, when ``ntrans>1`` a large amount of text can be generated.
    
- To diagnose problems with the spread/interpolation stage, similarly setting ``opts.spread_debug`` to 1 or 2 will print even more output. Here the setting 2 generates a large amount of output even for a single transform.



  
Other known issues with library or interfaces
**********************************************

The master list is the github issues for the project page,
https://github.com/flatironinstitute/finufft/issues.

A secondary and more speculative list is in the ``TODO`` text file.

Please look through those issue topics, since sometimes workarounds
are discussed before the problem is fixed in a release.



Bug reports
***********
  
If you think you have found a new bug, and have read the above, please
file a new issue on the github project page,
https://github.com/flatironinstitute/finufft/issues.
Include a minimal code which reproduces the bug, along with
details about your machine, operating system, compiler, version of FINUFFT, and output with ``opts.debug=2``.
If you have a known bug and have ideas, please add to the comments for that issue.

You may also contact Alex Barnett (``abarnett``
at-sign ``flatironinstitute.org``) with FINUFFT in the subject line.
