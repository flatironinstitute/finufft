.. _trouble:

Troubleshooting
===============

If you are having issues (segfaults, slowness, "wrong" answers, etc),
there is a high probability it is something we already know about, so
please first read all of the advice below in the section relevant
to your problem: math, speed, or crashing. Also look for
similar GitHub `Issues <https://github.com/flatironinstitute/finufft/issues?q=is%3Aissue>`_ or `Discussions <https://github.com/flatironinstitute/finufft/discussions>`_.
If that fails, post an Issue. The lead developer may also be contacted at abarnett@flatironinstitute.org


Mathematical "issues" and error size
************************************

- The type 1 and type 2 transforms are adjoints but **not inverses of each other** (unlike in the plain FFT case, where, up to a constant factor $N$, the adjoint is the inverse). Thus you cannot "undo" one transform by applying the other! (Unless the nonuniform points are precisely regularly spaced, and then you should instead be using the plain FFT). Therefore, if you are not getting the expected answers, please check that you have not made this assumption. In the :ref:`tutorials <tut>` we have an example showing how to invert the NUFFT; also see `NFFT3 inverse transforms <https://www-user.tu-chemnitz.de/~potts/nfft/infft.php>`_.

- If you request a tolerance ``tol`` that FINUFFT knows it cannot achieve, it will return the warning code ``ier=1`` after performing transforms as accurately as it can. Conversely, the status ``ier=0`` does not imply that the requested accuracy *was* achieved, merely that parameters were chosen to attempt this estimated accuracy. As our SISC paper shows, for typical situations, relative $\ell_2$ errors match the requested tolerances over a wide range, barring the caveats below. Users should always check *convergence* (by, for instance, varying ``tol`` and measuring any changes in their results); this is of course generally true in scientific computing.

- When requested tolerance ``tol`` is around $10^{-14}$ or less in double-precision, or $10^{-6}$ or less in single-precision, it will most likely be impossible for FINUFFT (or any other NUFFT library) to achieve this, due to inevitable round-off error. The next point goes into this in more detail.

- The **condition number of the problem** may be the factor limiting the accuracy in your application. In this case, *no NUFFT algorithm nor code could, even in principle, compute it more accurately* given the working precision of the input data! This applies particularly to 1D transforms with large $N$ (number of modes), say $N\ge 10^5$, and especially in single precision. Recall that machine error $\epsilon_{mach}$ is around ``6e-8`` in single and ``1e-16`` in double precision. The rule of thumb here is that one cannot demand that NUFFT relative output errors be smaller than $N_{max} \epsilon_{mach}$, where $N_{max}$ is the largest of the mode sizes $N_1,\dots,N_d$ in the $d$-dimensional case. This applies in the $\ell_2$ or maximum norms. No such dependence on $M$ occurs. In type 3 transforms, the $N_{max}$ should be replaced by the space-bandwidth product (maximum $x$ spread times maximum $k$ spread) in any dimension.

  Let us explain why, recalling the definition :eq:`1d1`. The simple reason is that $(d/dx_j) e^{ikx_j} = ik e^{ikx_j}$, so that the Jacobian derivative of the outputs of a type 1 or type 2 NUFFT, with respect to variation of the input locations $x_j$, grows like the mode index $k$. The magnitude of the latter can be as large as $N/2$, i.e., $O(N)$, or $O(N_{max})$ in the multi-dimensional case. Since the inputs $x_j$ inevitably have a rounding error of $\epsilon_{mach}$, this gets amplified by the above factor to give a lower bound on the error of even the best (most stable) algorithm for the NUFFT.

  In contrast the DFT (e.g., as computed by the FFT) has no growth in condition number vs $N$ (transform size), because it is (up to scaling) an isometry. The crucial difference for the NUFFT is the presence of the new input type (nonuniform point locations), to which it can be highly sensitive.

  For background on condition number of a problem, please read Ch. 12-15 of *Numerical Linear Algebra* by Trefethen and Bau (SIAM, 1997).

  NUFFT error is analysed and measured in Section 4.2 of our `SISC paper <https://arxiv.org/abs/1808.06736>`_, although in that work we omitted attributing the round-off error to the condition number of the problem.

- Finally, re error: while the tolerance ``tol`` is usually a good upper bound for observed accuracy (barring the $\epsilon_{mach}N_{max}$ error above), strangely, in single-precision requesting tolerance of $10^{-7}$ or $10^{-6}$ can give slightly *worse* accuracy than $10^{-5}$ or $10^{-4}$. We are looking into why. It is usually best to request at least 2--3 digits less accurate than the respective machine precision, for either single or double precision. Ie, do not set ``tol`` below $10^{-5}$ in single or $10^{-13}$ in double.


Speed issues and advice
***********************

CPU library speed
-----------------

If FINUFFT is slow (eg, less than $10^6$ to $10^7$ nonuniform points per second, depending on application), here is some advice:

- Try printing debug output to see step-by-step timings. Do this by setting ``opts.debug`` to 1 or 2 then looking at the timing information in stdout.

- Check that our test codes give similar speeds to what you observe, for a similar problem size. For example, if your application uses a 2D type 1 transform from a million nonuniform points to 500-by-500 modes, at 5-digit accuracy, using 8 threads, then build the tests and run::

    OMP_NUMTHREADS=8 test/finufft2d_test 500 500 1e6 1e-5

  which will give something like (on a laptop)::

    test 2d type 1:
	1000000 NU pts to (500,500) modes in 0.0403 s 	2.48e+07 NU pts/s
	one mode: rel err in F[185,130] is 4.38e-07
    test 2d type 2:
	(500,500) modes to 1000000 NU pts in 0.0274 s 	3.65e+07 NU pts/s
	one targ: rel err in c[500000] is 6.1e-07
    test 2d type 3:
	1000000 NU to 250000 NU in 0.0626 s         	2e+07 tot NU pts/s
	one targ: rel err in F[125000] is 2.76e-06

  Extract the relevant transform type (all three types are included), and compare its timing and throughput to your own. Usually the fact that these tests use random NU point distributions does not affect the speed that much compared to typical applications.
  If you instead use the vectorized ("many") interface for a stack of, say, 50 such transforms, use::

    OMP_NUMTHREADS=8 test/finufft2d_test 500 500 1e6 1e-5

  which compares the stack of transforms to the same transforms performed individually. For single precision tests, append ``f`` to the executable name in both of the above examples. The command line options for each tester can be seen by executing without any options.

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

- Are you calling the simple interface a huge number of times for small problems, but these tasks have something in common (number of modes, or locations of nonuniform points)? If so, try the vectorized or guru interfaces, which remove overheads in repeated FFTW plan look-up, and in bin-sorting. They can be 10-100x faster.

GPU library speed
-----------------

If cuFINUFFT is slow (eg, less than $10^8$ nonuniform points per second), here is some advice:

- Run our test codes with a similar problem size on your hardware. Build the tests, then, for example (matching the vectorized CPU example above)::

    test/cuda/cufinufft2dmany_test 1 1 500 500 50 0 1000000 1e-5 1e-4 f

  which gives (on my A6000) the output::

    #modes = 250000, #inputs = 50, #NUpts = 1000000
    [time  ] dummy warmup call to CUFFT	 0.00184 s
    [time  ] cufinufft plan:		 0.000624 s
    [time  ] cufinufft setNUpts:         0.000431 s
    [time  ] cufinufft exec:		 0.0839 s
    [time  ] cufinufft destroy:		 0.00194 s
    [gpu   ] 49th data one mode: rel err in F[185,130] is 2.61e-05
    [totaltime] 8.69e+04 us, speed 5.76e+08 NUpts/s
					(exec-only thoughput: 5.96e+08 NU pts/s)

  Check if your time is dominated by the plan stage, and if so, try to reuse your plan (often one has repeated transforms with sizes or points in common). Sometimes the CUFFT warm-up call can take as long as 0.2 seconds; make sure you do such a call (or a dummy transform) before your timed usage occurs. See https://github.com/flatironinstitute/finufft/issues/385 for an example of this discovery process. The command line options for each tester can be seen by executing without any options. Note that ``1e6`` for the GPU testers is not interpreted as $10^6$, unlike in the CPU testers.

- Try the different method types. Start with method=1. For instance, for type 1 transforms, method 2 (SM in the paper) is supposed to be faster than method 1 (GM-sort in the paper), but on the above test it is only 2% faster. In the test call, the 1st argument sets the method type and the next argument the transform type.

- There is not currently a ``debug`` option for ``cufinufft``, so the above timing of a test problem on your hardware is a good option. You could place timers around the various ``cufinufft`` calls in your own code, just as in our test codes.


  
Crash (segfault) issues and advice
****************************************

- Are you using ``int64`` (``integer*8``) types for sizes ``M``, ``N``, etc? (If you have warnings switched off, you may not notice this until execution.)

- Are you passing in pointers to the wrong size of object, eg, single vs double precision? The library includes both precisions, so make sure you are calling the correct one (commands begin ``finufft`` for double, ``finufftf`` for single).

- If you use C++/C/Fortran and changed the options struct values, did you forget to call ``finufft_default_opts`` first?

- Maybe you have switched off nonuniform point bounds checking (``opts.chkbnds=0``) for a little extra speed? Try switching it on again to catch illegal coordinates.

- Thread-safety: are you calling FINUFFT from inside a multithreaded block of code without setting ``opts.nthreads=1``? If ``gdb`` indicates crashes during FFTW calls, this is another sign.
  
- To isolate where a crash is occurring, set ``opts.debug`` to 1 or 2, and check the text output of the various stages. With a debug setting of 2 or above, when ``ntrans>1`` a large amount of text can be generated.
    
- To diagnose problems with the spread/interpolation stage, similarly setting ``opts.spread_debug`` to 1 or 2 will print even more output. Here the setting 2 generates a large amount of output even for a single transform.

- For the GPU code, did you run out of GPU memory? Keep track of this with ``nvidia-smi``.

  
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
