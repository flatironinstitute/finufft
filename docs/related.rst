.. _related:

Related packages
================

Other recommended NUFFT libraries
---------------------------------

- `NFFT3 <https://www-user.tu-chemnitz.de/~potts/nfft/>`__: well-supported and multi-featured C++ library using FFTW. Has MATLAB MEX interface. However, NFFT is significantly slower and/or more memory-intensive than FINUFFT (see reference [FIN], and our :doc:`migration guide <nfft_migr>`). NFFT3 has more general abilities, eg, inverse NUFFTs.

- `CMCL NUFFT <https://cims.nyu.edu/cmcl/nufft/nufft.html>`__: NYU single-threaded Fortran library using self-contained FFT, fast Gaussian gridding kernel. Has MATLAB MEX interface. Much (up to 50x even for one thread) slower than FINUFFT, but is very easy to compile.

- `MIRT <https://web.eecs.umich.edu/~fessler/code/index.html>`__ Michigan Image Reconstruction Toolbox. Native MATLAB, single-threaded sparse mat-vec, prestores all kernel evaluations, thus is memory-intensive but surprisingly fast for a single-threaded implementation. However, slower than FINUFFT for all tolerances smaller than $10^{-1}$.

- `PyNUFFT <https://github.com/jyhmiinlin/pynufft>`__ Python code supporting CPU and GPU operation. We have not compared against FINUFFT yet.

- `NonuniformFFTs.jl <https://jipolanco.github.io/NonuniformFFTs.jl/dev/>`__ native Julia code for types 1 and 2 only (CPU and GPU via KernelAbstractions), by Juan Polanco, 2024. Close to our CPU performance, and can beat it in the case of real data via a custom real transform. On the GPU claims their shared-memory type 1 implementation beats our v2.4.1. Has a good `benchmarks page <https://jipolanco.github.io/NonuniformFFTs.jl/dev/benchmarks/>`__ comparing (cu)FINUFFT at 6-digit accuracy, CPU and GPU. Marco has now incorporated his ideas into output-driven type 1 on GPU (for v2.5.0).

- `NFFT.jl <https://github.com/JuliaMath/NFFT.jl>`__ native Julia implementation for type 1 and 2 only, by Tobias Knopp and coworkers, starting around 2022. See :doc:`page on Julia <julia>`.

A comparison of some library performances (as of 2019) was in our paper [FIN] in the :doc:`references <refs>`.
