Related packages
================

Other recommended NUFFT libraries
---------------------------------

- `NFFT3 <https://www-user.tu-chemnitz.de/~potts/nfft/>`_: well-supported and multi-featured C++ library using FFTW. Has MATLAB interface. However, significantly slower and/or more memory-intensive than FINUFFT (see reference [FIN]).

- `CMCL NUFFT <https://cims.nyu.edu/cmcl/nufft/nufft.html>`_: NYU single-threaded Fortran library using self-contained FFT, Gaussian kernel. Has MATLAB interface. Much slower than FINUFFT.

- `MIRT <https://web.eecs.umich.edu/~fessler/code/index.html>`_ Michigan Image Reconstruction Toolbox. Native MATLAB, single-threaded sparse mat-vec, prestores all kernel evaluations, thus is memory-intensive. Slower than FINUFFT for all tolerances smaller than 0.1.

- `PyNUFFT <https://github.com/jyhmiinlin/pynufft>`_ Python code supporting CPU and GPU operation. Have not compared against FINUFFT yet.

  
Also see the summary of library performances in our paper [FIN].
  

Interfaces to FINUFFT from other languages
------------------------------------------

- `FINUFFT.jl <https://github.com/ludvigak/FINUFFT.jl>`_: a `julia <https://julialang.org/>`_ language wrapper by Ludvig af Klinteberg (SFU). This is actually a secondary wrapper around our python interface, so you should make sure that the latter is working first.

- Vineet Bansal's pypi package https://pypi.org/project/finufftpy/


Packages making use of FINUFFT
------------------------------

Here are some packages dependent on FINUFFT (please let us know others):

- `ASPIRE <http://spr.math.princeton.edu>`_: software for cryo-EM, based at Amit Singer's group at Princeton. `github <https://github.com/PrincetonUniversity/ASPIRE-Python>`_

- `sinctransform <https://github.com/hannahlawrence/sinctransform>`_: C++
  and MATLAB codes to evaluate sums of the sinc and sinc^2 kernels between arbitrary nonuniform points in 1,2, or 3 dimensions, by Hannah Lawrence (2017 summer intern at Flatiron).
