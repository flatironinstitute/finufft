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

CI generates the performance data. This section will populate after the
perftest workflow runs on master.

GPU
---

CI generates the performance data. This section will populate after the
perftest workflow runs on master.
