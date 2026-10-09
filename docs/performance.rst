.. _performance:

Performance
===========

This page compares measured performance across FINUFFT releases and the latest commit on the master branch.
One goal is to document progress between releases. Another goal is to ensure that performance does not regress.
Users unsure about the performance on their machine should compare their timings against these results. The :ref:`troubleshooting <trouble>` page gives further advice.
The results can also guide the compile-time configuration (compiler, flags, FFT library) and the runtime parameters (upsampling factor, number of threads).

CPU performance depends on the problem: dimensions, size, transform type, and requested accuracy.
CPU performance also depends on the measurement setup: upsampling factor, number of threads, compiler flags, available SIMD instructions, and (since 2.3.0) the FFT library.
The curse of dimensionality prevents testing every combination, so the cases below are user scenarios selected from this `GitHub discussion <https://github.com/flatironinstitute/finufft/discussions/398>`__.
If no case covers a given use case, comment in that discussion and the benchmark set can be extended.
This `GitHub discussion <https://github.com/flatironinstitute/finufft/discussions/452>`__ benchmarks the spreader/interpolator alone under different compilers and indicates which compiler is fastest for a specific CPU.

Each graph stacks the minimum duration of each stage per version: makeplan, setpts, and execute.
The minimum is the fastest of repeated runs of a case, the measurement least polluted by machine noise.
The speedup label above a version, for example ``1.10x``, states the factor by which the version is faster than the baseline.
The baseline is the leftmost version: the oldest release, or master in pull-request comparisons.

A Jenkins job regenerates this page on every push to master.
The job recompiles every library version with the ``cmake`` flags ``-DFINUFFT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release`` and measures all versions in a single run.
Every version therefore runs with the same compiler on the same CPU.
The exact CPU model and compiler version depend on the node the job lands on. Each backend heading lists the measured hardware and the compiler.

The page has one section per library.
The CPU section covers FINUFFT with its two FFT backends, FFTW and DUCC.
One job measures both backends back to back on one CPU. The two backends therefore share the measurement conditions and the parameters.
In FFT-bound problems, DUCC is expected to outperform FFTW in 2D and 3D. In 1D, FFTW is expected to be faster.
The GPU section covers cuFINUFFT. A separate job measures cuFINUFFT on one card, on the same case list as the CPU section minus the CPU-only thread count.
Each section groups the plots by transform type and dimensionality.

.. contents:: On this page
   :local:
   :depth: 2

.. PERFTEST_BACKENDS_BELOW


CPU
---


FFTW backend
~~~~~~~~~~~~

CPU: ``Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz``.

Arch: ``X86_64``.

Usable processors: ``29``.

Usable physical cores: ``29``.

Microarchitecture: ``skylake_avx512``.

psABI level: ``x86-64-v4``.

Compiler: ``c++ (GCC) 13.3.1 20240611 (Red Hat 13.3.1-2)``.

Compiler flags: ``-march=native``.



1D transforms
^^^^^^^^^^^^^^^^^^^^^


Type 1
""""""""""""""""




Parameters: ``prec:f N1:1e4 N2:1 N3:1
ntransf:1 threads:1 M:1e7 tol:2e-3``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_03fddf32ebd6d1ad.svg
   :alt: pics/perftestci_03fddf32ebd6d1ad.svg
   :width: 100%


Parameters: ``prec:d N1:1e4 N2:1 N3:1
ntransf:1 threads:1 M:1e7 tol:1e-9``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_1a97e74700df8edf.svg
   :alt: pics/perftestci_1a97e74700df8edf.svg
   :width: 100%




Type 2
""""""""""""""""




Parameters: ``prec:f N1:1e4 N2:1 N3:1
ntransf:1 threads:1 M:1e7 tol:2e-3``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_32470af6745a29e6.svg
   :alt: pics/perftestci_32470af6745a29e6.svg
   :width: 100%


Parameters: ``prec:d N1:1e4 N2:1 N3:1
ntransf:1 threads:1 M:1e7 tol:1e-9``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_eab5fef5bbe3e744.svg
   :alt: pics/perftestci_eab5fef5bbe3e744.svg
   :width: 100%




Type 3
""""""""""""""""




Parameters: ``prec:f N1:1e4 N2:1 N3:1
ntransf:1 threads:1 M:1e7 tol:2e-3``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_0699064210be24b8.svg
   :alt: pics/perftestci_0699064210be24b8.svg
   :width: 100%


Parameters: ``prec:d N1:1e4 N2:1 N3:1
ntransf:1 threads:1 M:1e7 tol:1e-9``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_9db50339e1439054.svg
   :alt: pics/perftestci_9db50339e1439054.svg
   :width: 100%





2D transforms
^^^^^^^^^^^^^^^^^^^^^


Type 1
""""""""""""""""




Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 threads:1 M:1e7 tol:1e-4``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_77ba4ad1680da735.svg
   :alt: pics/perftestci_77ba4ad1680da735.svg
   :width: 100%


Parameters: ``prec:d N1:320 N2:320 N3:1
ntransf:1 threads:1 M:1e7 tol:1e-9``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_33c98f79a3ac719a.svg
   :alt: pics/perftestci_33c98f79a3ac719a.svg
   :width: 100%


Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 threads:0 M:3e5 tol:1e-4``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_9625f1c99c72af93.svg
   :alt: pics/perftestci_9625f1c99c72af93.svg
   :width: 100%




Type 2
""""""""""""""""




Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 threads:1 M:1e7 tol:1e-4``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_eb3a21326e672e21.svg
   :alt: pics/perftestci_eb3a21326e672e21.svg
   :width: 100%


Parameters: ``prec:d N1:320 N2:320 N3:1
ntransf:1 threads:1 M:1e7 tol:1e-9``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_da952c8b20d25f6d.svg
   :alt: pics/perftestci_da952c8b20d25f6d.svg
   :width: 100%


Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 threads:0 M:3e5 tol:1e-4``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_d9ae761d8c6edf34.svg
   :alt: pics/perftestci_d9ae761d8c6edf34.svg
   :width: 100%




Type 3
""""""""""""""""




Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 threads:1 M:1e7 tol:1e-4``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_5c3bb7a151aa63c1.svg
   :alt: pics/perftestci_5c3bb7a151aa63c1.svg
   :width: 100%


Parameters: ``prec:d N1:320 N2:320 N3:1
ntransf:1 threads:1 M:1e7 tol:1e-9``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_c8548db3f2e8e4bd.svg
   :alt: pics/perftestci_c8548db3f2e8e4bd.svg
   :width: 100%


Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 threads:0 M:3e5 tol:1e-4``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_588b1a3ca8008a80.svg
   :alt: pics/perftestci_588b1a3ca8008a80.svg
   :width: 100%





3D transforms
^^^^^^^^^^^^^^^^^^^^^


Type 1
""""""""""""""""




Parameters: ``prec:d N1:192 N2:192 N3:128
ntransf:1 threads:0 M:8e4 tol:1e-7``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_8d6939a8faa801ae.svg
   :alt: pics/perftestci_8d6939a8faa801ae.svg
   :width: 100%




Type 2
""""""""""""""""




Parameters: ``prec:d N1:192 N2:192 N3:128
ntransf:1 threads:0 M:8e4 tol:1e-7``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_261b0882ecf2322a.svg
   :alt: pics/perftestci_261b0882ecf2322a.svg
   :width: 100%




Type 3
""""""""""""""""




Parameters: ``prec:d N1:192 N2:192 N3:128
ntransf:1 threads:0 M:8e4 tol:1e-7``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_50b8df8c5589afe8.svg
   :alt: pics/perftestci_50b8df8c5589afe8.svg
   :width: 100%


DUCC backend
~~~~~~~~~~~~

CPU: ``Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz``.

Arch: ``X86_64``.

Usable processors: ``29``.

Usable physical cores: ``29``.

Microarchitecture: ``skylake_avx512``.

psABI level: ``x86-64-v4``.

Compiler: ``c++ (GCC) 13.3.1 20240611 (Red Hat 13.3.1-2)``.

Compiler flags: ``-march=native``.



1D transforms
^^^^^^^^^^^^^^^^^^^^^


Type 1
""""""""""""""""




Parameters: ``prec:f N1:1e4 N2:1 N3:1
ntransf:1 threads:1 M:1e7 tol:2e-3``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_aa552f73f4b4949b.svg
   :alt: pics/perftestci_aa552f73f4b4949b.svg
   :width: 100%


Parameters: ``prec:d N1:1e4 N2:1 N3:1
ntransf:1 threads:1 M:1e7 tol:1e-9``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_8b7238c4c23685d2.svg
   :alt: pics/perftestci_8b7238c4c23685d2.svg
   :width: 100%




Type 2
""""""""""""""""




Parameters: ``prec:f N1:1e4 N2:1 N3:1
ntransf:1 threads:1 M:1e7 tol:2e-3``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_02faa6e342257b82.svg
   :alt: pics/perftestci_02faa6e342257b82.svg
   :width: 100%


Parameters: ``prec:d N1:1e4 N2:1 N3:1
ntransf:1 threads:1 M:1e7 tol:1e-9``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_67000b762fce7ccd.svg
   :alt: pics/perftestci_67000b762fce7ccd.svg
   :width: 100%




Type 3
""""""""""""""""




Parameters: ``prec:f N1:1e4 N2:1 N3:1
ntransf:1 threads:1 M:1e7 tol:2e-3``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_a731766cdd9c953d.svg
   :alt: pics/perftestci_a731766cdd9c953d.svg
   :width: 100%


Parameters: ``prec:d N1:1e4 N2:1 N3:1
ntransf:1 threads:1 M:1e7 tol:1e-9``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_7e8bc5fc24054781.svg
   :alt: pics/perftestci_7e8bc5fc24054781.svg
   :width: 100%





2D transforms
^^^^^^^^^^^^^^^^^^^^^


Type 1
""""""""""""""""




Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 threads:1 M:1e7 tol:1e-4``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_776af2ee5f010040.svg
   :alt: pics/perftestci_776af2ee5f010040.svg
   :width: 100%


Parameters: ``prec:d N1:320 N2:320 N3:1
ntransf:1 threads:1 M:1e7 tol:1e-9``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_6f4b53fdcb30c836.svg
   :alt: pics/perftestci_6f4b53fdcb30c836.svg
   :width: 100%


Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 threads:0 M:3e5 tol:1e-4``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_1772b7801308a5e4.svg
   :alt: pics/perftestci_1772b7801308a5e4.svg
   :width: 100%




Type 2
""""""""""""""""




Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 threads:1 M:1e7 tol:1e-4``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_be36c150e3dcc402.svg
   :alt: pics/perftestci_be36c150e3dcc402.svg
   :width: 100%


Parameters: ``prec:d N1:320 N2:320 N3:1
ntransf:1 threads:1 M:1e7 tol:1e-9``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_6956b981d30ffa82.svg
   :alt: pics/perftestci_6956b981d30ffa82.svg
   :width: 100%


Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 threads:0 M:3e5 tol:1e-4``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_1ca39b51ccf401d5.svg
   :alt: pics/perftestci_1ca39b51ccf401d5.svg
   :width: 100%




Type 3
""""""""""""""""




Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 threads:1 M:1e7 tol:1e-4``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_e84a28c7935892f9.svg
   :alt: pics/perftestci_e84a28c7935892f9.svg
   :width: 100%


Parameters: ``prec:d N1:320 N2:320 N3:1
ntransf:1 threads:1 M:1e7 tol:1e-9``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_c6bc9ac2b5ee9235.svg
   :alt: pics/perftestci_c6bc9ac2b5ee9235.svg
   :width: 100%


Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 threads:0 M:3e5 tol:1e-4``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_9458ccb8da322d03.svg
   :alt: pics/perftestci_9458ccb8da322d03.svg
   :width: 100%





3D transforms
^^^^^^^^^^^^^^^^^^^^^


Type 1
""""""""""""""""




Parameters: ``prec:d N1:192 N2:192 N3:128
ntransf:1 threads:0 M:8e4 tol:1e-7``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_d47b1c94b158fe17.svg
   :alt: pics/perftestci_d47b1c94b158fe17.svg
   :width: 100%




Type 2
""""""""""""""""




Parameters: ``prec:d N1:192 N2:192 N3:128
ntransf:1 threads:0 M:8e4 tol:1e-7``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_f358d79bfbdd7df0.svg
   :alt: pics/perftestci_f358d79bfbdd7df0.svg
   :width: 100%




Type 3
""""""""""""""""




Parameters: ``prec:d N1:192 N2:192 N3:128
ntransf:1 threads:0 M:8e4 tol:1e-7``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_f2e70390ce49da03.svg
   :alt: pics/perftestci_f2e70390ce49da03.svg
   :width: 100%


GPU
---


cuFFT backend
~~~~~~~~~~~~~

Device: ``Tesla V100-PCIE-16GB, 7.0, 16384 MiB, 580.178.04``.

Toolkit: ``Cuda compilation tools, release 12.8, V12.8.93``.

Host compiler: ``c++ (GCC) 13.3.1 20240611 (Red Hat 13.3.1-2)``.



1D transforms
^^^^^^^^^^^^^^^^^^^^^


Type 1
""""""""""""""""



Method 0 (automatic choice)
+++++++++++++++++++++++++++



Parameters: ``prec:f N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:2e-3 method:0``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_c249f9e4a0229945.svg
   :alt: pics/perftestci_c249f9e4a0229945.svg
   :width: 100%


Parameters: ``prec:d N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:1e-9 method:0``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_1610c2d6471cb3e6.svg
   :alt: pics/perftestci_1610c2d6471cb3e6.svg
   :width: 100%




Method 1 (GM, points-driven)
++++++++++++++++++++++++++++



Parameters: ``prec:f N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:2e-3 method:1``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_c2caef58f7f9a813.svg
   :alt: pics/perftestci_c2caef58f7f9a813.svg
   :width: 100%


Parameters: ``prec:d N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:1e-9 method:1``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_c7ae4621acb32068.svg
   :alt: pics/perftestci_c7ae4621acb32068.svg
   :width: 100%




Method 2 (SM, subproblem)
+++++++++++++++++++++++++



Parameters: ``prec:f N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:2e-3 method:2``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_2aaf2ceaac4f22a7.svg
   :alt: pics/perftestci_2aaf2ceaac4f22a7.svg
   :width: 100%


Parameters: ``prec:d N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:1e-9 method:2``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_c3219093d4de8bc3.svg
   :alt: pics/perftestci_c3219093d4de8bc3.svg
   :width: 100%




Method 3 (OD, output-driven)
++++++++++++++++++++++++++++



Parameters: ``prec:f N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:2e-3 method:3``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_095bbdd550d0bcf8.svg
   :alt: pics/perftestci_095bbdd550d0bcf8.svg
   :width: 100%


Parameters: ``prec:d N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:1e-9 method:3``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_9262b5c014206647.svg
   :alt: pics/perftestci_9262b5c014206647.svg
   :width: 100%




Type 2
""""""""""""""""



Method 0 (automatic choice)
+++++++++++++++++++++++++++



Parameters: ``prec:f N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:2e-3 method:0``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_421b58ae8120dcf1.svg
   :alt: pics/perftestci_421b58ae8120dcf1.svg
   :width: 100%


Parameters: ``prec:d N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:1e-9 method:0``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_2ba53440a02062e3.svg
   :alt: pics/perftestci_2ba53440a02062e3.svg
   :width: 100%




Method 1 (GM, points-driven)
++++++++++++++++++++++++++++



Parameters: ``prec:f N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:2e-3 method:1``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_fad4bde40df91114.svg
   :alt: pics/perftestci_fad4bde40df91114.svg
   :width: 100%


Parameters: ``prec:d N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:1e-9 method:1``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_f0966d306c0130ad.svg
   :alt: pics/perftestci_f0966d306c0130ad.svg
   :width: 100%




Type 3
""""""""""""""""



Method 0 (automatic choice)
+++++++++++++++++++++++++++



Parameters: ``prec:f N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:2e-3 method:0``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_3e0df55dd9129f2a.svg
   :alt: pics/perftestci_3e0df55dd9129f2a.svg
   :width: 100%


Parameters: ``prec:d N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:1e-9 method:0``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_c6320d1bf9841cd5.svg
   :alt: pics/perftestci_c6320d1bf9841cd5.svg
   :width: 100%




Method 1 (GM, points-driven)
++++++++++++++++++++++++++++



Parameters: ``prec:f N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:2e-3 method:1``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_b95c3a0431bd6fa8.svg
   :alt: pics/perftestci_b95c3a0431bd6fa8.svg
   :width: 100%


Parameters: ``prec:d N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:1e-9 method:1``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_c34cd15fa8daae05.svg
   :alt: pics/perftestci_c34cd15fa8daae05.svg
   :width: 100%




Method 2 (SM, subproblem)
+++++++++++++++++++++++++



Parameters: ``prec:f N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:2e-3 method:2``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_0b7dada261ead35d.svg
   :alt: pics/perftestci_0b7dada261ead35d.svg
   :width: 100%


Parameters: ``prec:d N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:1e-9 method:2``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_fc5d40f52302400d.svg
   :alt: pics/perftestci_fc5d40f52302400d.svg
   :width: 100%




Method 3 (OD, output-driven)
++++++++++++++++++++++++++++



Parameters: ``prec:f N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:2e-3 method:3``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_e6e056901aa3e509.svg
   :alt: pics/perftestci_e6e056901aa3e509.svg
   :width: 100%


Parameters: ``prec:d N1:1e4 N2:1 N3:1
ntransf:1 M:1e7 tol:1e-9 method:3``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_01009835b2d04ef1.svg
   :alt: pics/perftestci_01009835b2d04ef1.svg
   :width: 100%





2D transforms
^^^^^^^^^^^^^^^^^^^^^


Type 1
""""""""""""""""



Method 0 (automatic choice)
+++++++++++++++++++++++++++



Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-4 method:0``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_4841429b42fa3709.svg
   :alt: pics/perftestci_4841429b42fa3709.svg
   :width: 100%


Parameters: ``prec:d N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-9 method:0``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_a0532e7c7aed78a3.svg
   :alt: pics/perftestci_a0532e7c7aed78a3.svg
   :width: 100%


Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:3e5 tol:1e-4 method:0``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_bd4e941b91e9a6c0.svg
   :alt: pics/perftestci_bd4e941b91e9a6c0.svg
   :width: 100%




Method 1 (GM, points-driven)
++++++++++++++++++++++++++++



Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-4 method:1``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_9b4a4a2c4046daa9.svg
   :alt: pics/perftestci_9b4a4a2c4046daa9.svg
   :width: 100%


Parameters: ``prec:d N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-9 method:1``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_3bfffd149bf463ef.svg
   :alt: pics/perftestci_3bfffd149bf463ef.svg
   :width: 100%


Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:3e5 tol:1e-4 method:1``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_2aac3d045fd72872.svg
   :alt: pics/perftestci_2aac3d045fd72872.svg
   :width: 100%




Method 2 (SM, subproblem)
+++++++++++++++++++++++++



Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-4 method:2``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_ceca4d66ee9eaf0c.svg
   :alt: pics/perftestci_ceca4d66ee9eaf0c.svg
   :width: 100%


Parameters: ``prec:d N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-9 method:2``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_44840b509703d91c.svg
   :alt: pics/perftestci_44840b509703d91c.svg
   :width: 100%


Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:3e5 tol:1e-4 method:2``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_e3f4aae308422bed.svg
   :alt: pics/perftestci_e3f4aae308422bed.svg
   :width: 100%




Method 3 (OD, output-driven)
++++++++++++++++++++++++++++



Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-4 method:3``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_dd2c7b251222ca46.svg
   :alt: pics/perftestci_dd2c7b251222ca46.svg
   :width: 100%


Parameters: ``prec:d N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-9 method:3``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_a0be5b88de39a68a.svg
   :alt: pics/perftestci_a0be5b88de39a68a.svg
   :width: 100%


Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:3e5 tol:1e-4 method:3``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_520c91690447ef2a.svg
   :alt: pics/perftestci_520c91690447ef2a.svg
   :width: 100%




Type 2
""""""""""""""""



Method 0 (automatic choice)
+++++++++++++++++++++++++++



Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-4 method:0``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_7f3bc272395e7fe2.svg
   :alt: pics/perftestci_7f3bc272395e7fe2.svg
   :width: 100%


Parameters: ``prec:d N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-9 method:0``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_79bf569a2a1e1c04.svg
   :alt: pics/perftestci_79bf569a2a1e1c04.svg
   :width: 100%


Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:3e5 tol:1e-4 method:0``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_240618c0b5a5a123.svg
   :alt: pics/perftestci_240618c0b5a5a123.svg
   :width: 100%




Method 1 (GM, points-driven)
++++++++++++++++++++++++++++



Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-4 method:1``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_ce0ad44aef08498d.svg
   :alt: pics/perftestci_ce0ad44aef08498d.svg
   :width: 100%


Parameters: ``prec:d N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-9 method:1``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_f6a034cfb9c780c4.svg
   :alt: pics/perftestci_f6a034cfb9c780c4.svg
   :width: 100%


Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:3e5 tol:1e-4 method:1``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_9b8a8aded67458b3.svg
   :alt: pics/perftestci_9b8a8aded67458b3.svg
   :width: 100%




Method 2 (SM, subproblem)
+++++++++++++++++++++++++



Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-4 method:2``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_1c8be845cb7d1dca.svg
   :alt: pics/perftestci_1c8be845cb7d1dca.svg
   :width: 100%


Parameters: ``prec:d N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-9 method:2``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_91b2197092ff45d2.svg
   :alt: pics/perftestci_91b2197092ff45d2.svg
   :width: 100%


Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:3e5 tol:1e-4 method:2``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_a04977875c3dc31a.svg
   :alt: pics/perftestci_a04977875c3dc31a.svg
   :width: 100%




Type 3
""""""""""""""""



Method 0 (automatic choice)
+++++++++++++++++++++++++++



Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-4 method:0``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_4e89860713dbbc33.svg
   :alt: pics/perftestci_4e89860713dbbc33.svg
   :width: 100%


Parameters: ``prec:d N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-9 method:0``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_27efae826d47077f.svg
   :alt: pics/perftestci_27efae826d47077f.svg
   :width: 100%


Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:3e5 tol:1e-4 method:0``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_c25ca7b4717c40d0.svg
   :alt: pics/perftestci_c25ca7b4717c40d0.svg
   :width: 100%




Method 1 (GM, points-driven)
++++++++++++++++++++++++++++



Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-4 method:1``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_aed25b197273328a.svg
   :alt: pics/perftestci_aed25b197273328a.svg
   :width: 100%


Parameters: ``prec:d N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-9 method:1``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_87cc148ea64f073f.svg
   :alt: pics/perftestci_87cc148ea64f073f.svg
   :width: 100%


Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:3e5 tol:1e-4 method:1``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_73aad291e659f350.svg
   :alt: pics/perftestci_73aad291e659f350.svg
   :width: 100%




Method 2 (SM, subproblem)
+++++++++++++++++++++++++



Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-4 method:2``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_8b2fd2b2524d696b.svg
   :alt: pics/perftestci_8b2fd2b2524d696b.svg
   :width: 100%


Parameters: ``prec:d N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-9 method:2``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_63d8890d10b255a2.svg
   :alt: pics/perftestci_63d8890d10b255a2.svg
   :width: 100%


Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:3e5 tol:1e-4 method:2``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_4a135e01cd5bac49.svg
   :alt: pics/perftestci_4a135e01cd5bac49.svg
   :width: 100%




Method 3 (OD, output-driven)
++++++++++++++++++++++++++++



Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-4 method:3``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_dea5c4fa1fbacce7.svg
   :alt: pics/perftestci_dea5c4fa1fbacce7.svg
   :width: 100%


Parameters: ``prec:d N1:320 N2:320 N3:1
ntransf:1 M:1e7 tol:1e-9 method:3``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_03ec61a5bf0c1f1d.svg
   :alt: pics/perftestci_03ec61a5bf0c1f1d.svg
   :width: 100%


Parameters: ``prec:f N1:320 N2:320 N3:1
ntransf:1 M:3e5 tol:1e-4 method:3``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_339845044f42f562.svg
   :alt: pics/perftestci_339845044f42f562.svg
   :width: 100%





3D transforms
^^^^^^^^^^^^^^^^^^^^^


Type 1
""""""""""""""""



Method 0 (automatic choice)
+++++++++++++++++++++++++++



Parameters: ``prec:d N1:192 N2:192 N3:128
ntransf:1 M:8e4 tol:1e-7 method:0``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_bf58f99b2f8308a0.svg
   :alt: pics/perftestci_bf58f99b2f8308a0.svg
   :width: 100%




Method 1 (GM, points-driven)
++++++++++++++++++++++++++++



Parameters: ``prec:d N1:192 N2:192 N3:128
ntransf:1 M:8e4 tol:1e-7 method:1``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_17736987bca3c6e9.svg
   :alt: pics/perftestci_17736987bca3c6e9.svg
   :width: 100%




Method 2 (SM, subproblem)
+++++++++++++++++++++++++



Parameters: ``prec:d N1:192 N2:192 N3:128
ntransf:1 M:8e4 tol:1e-7 method:2``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_5f96c33c37b217dd.svg
   :alt: pics/perftestci_5f96c33c37b217dd.svg
   :width: 100%




Method 3 (OD, output-driven)
++++++++++++++++++++++++++++



Parameters: ``prec:d N1:192 N2:192 N3:128
ntransf:1 M:8e4 tol:1e-7 method:3``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_55383a6b087f8e4f.svg
   :alt: pics/perftestci_55383a6b087f8e4f.svg
   :width: 100%




Type 2
""""""""""""""""



Method 0 (automatic choice)
+++++++++++++++++++++++++++



Parameters: ``prec:d N1:192 N2:192 N3:128
ntransf:1 M:8e4 tol:1e-7 method:0``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_a842d3b93695b2c3.svg
   :alt: pics/perftestci_a842d3b93695b2c3.svg
   :width: 100%




Method 1 (GM, points-driven)
++++++++++++++++++++++++++++



Parameters: ``prec:d N1:192 N2:192 N3:128
ntransf:1 M:8e4 tol:1e-7 method:1``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_e25c46e72df283a1.svg
   :alt: pics/perftestci_e25c46e72df283a1.svg
   :width: 100%




Method 2 (SM, subproblem)
+++++++++++++++++++++++++



Parameters: ``prec:d N1:192 N2:192 N3:128
ntransf:1 M:8e4 tol:1e-7 method:2``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_6d65c65ea63b4ada.svg
   :alt: pics/perftestci_6d65c65ea63b4ada.svg
   :width: 100%




Type 3
""""""""""""""""



Method 0 (automatic choice)
+++++++++++++++++++++++++++



Parameters: ``prec:d N1:192 N2:192 N3:128
ntransf:1 M:8e4 tol:1e-7 method:0``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_cf989dee69ce538a.svg
   :alt: pics/perftestci_cf989dee69ce538a.svg
   :width: 100%




Method 1 (GM, points-driven)
++++++++++++++++++++++++++++



Parameters: ``prec:d N1:192 N2:192 N3:128
ntransf:1 M:8e4 tol:1e-7 method:1``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_0b19339ecc8340e4.svg
   :alt: pics/perftestci_0b19339ecc8340e4.svg
   :width: 100%




Method 2 (SM, subproblem)
+++++++++++++++++++++++++



Parameters: ``prec:d N1:192 N2:192 N3:128
ntransf:1 M:8e4 tol:1e-7 method:2``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_6216b3534b07829a.svg
   :alt: pics/perftestci_6216b3534b07829a.svg
   :width: 100%




Method 3 (OD, output-driven)
++++++++++++++++++++++++++++



Parameters: ``prec:d N1:192 N2:192 N3:128
ntransf:1 M:8e4 tol:1e-7 method:3``

.. image:: https://raw.githubusercontent.com/flatironinstitute/finufft/perftest-results/docs/pics/perftestci_fc9cb01b2fc57224.svg
   :alt: pics/perftestci_fc9cb01b2fc57224.svg
   :width: 100%

