.. _related:

Related packages
================

Other recommended NUFFT libraries
---------------------------------

- `cuFINUFFT <https://github.com/flatironinstitute/cufinufft>`_: Our GPU version of FINUFFT, for single precision in 2D and 3D, type 1 and 2. Still under development by Melody Shih (NYU) and others. Often achieves speeds around 10x the CPU version.

- `NFFT3 <https://www-user.tu-chemnitz.de/~potts/nfft/>`_: well-supported and multi-featured C++ library using FFTW. Has MATLAB MEX interface. However, significantly slower and/or more memory-intensive than FINUFFT (see reference [FIN]). Has many more general abilities, eg, inverse NUFFT. We are working on this too.

- `CMCL NUFFT <https://cims.nyu.edu/cmcl/nufft/nufft.html>`_: NYU single-threaded Fortran library using self-contained FFT, fast Gaussian gridding kernel. Has MATLAB MEX interface. Much (up to 50x even for one thread) slower than FINUFFT, but is very easy to compile.

- `MIRT <https://web.eecs.umich.edu/~fessler/code/index.html>`_ Michigan Image Reconstruction Toolbox. Native MATLAB, single-threaded sparse mat-vec, prestores all kernel evaluations, thus is memory-intensive but surprisingly fast for a single-threaded implementation. However, slower than FINUFFT for all tolerances smaller than $10^{-1}$.

- `PyNUFFT <https://github.com/jyhmiinlin/pynufft>`_ Python code supporting CPU and GPU operation. We have not compared against FINUFFT yet.

  
Also see the summary of library performances in our paper [FIN] in the
:ref:`references <refs>`.
  
