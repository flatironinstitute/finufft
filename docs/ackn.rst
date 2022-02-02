.. _ackn:

Acknowledgments
===============

FINUFFT was initiated by Jeremy Magland and Alex Barnett at the
Center for Computational Mathematics, Flatiron Institute in early 2017.
The main developer and maintainer is:

* Alex Barnett

Major code contributions by:

* Jeremy Magland - early multithreaded spreader, benchmark vs other libraries, py wrapper
* Ludvig af Klinteberg - SIMD vectorization/acceleration of spreader, julia wrapper
* Yu-Hsuan ("Melody") Shih - 2d1many, 2d2many vectorized interface, GPU version
* Andrea Malleo - guru interface prototype and tests
* Libin Lu - guru Fortran, python, MATLAB/octave, julia interfaces
* Joakim Andén - python, MATLAB/FFTW issues, dual-precision, performance tests

Other significant code contributions by:

* Robert Blackwell - atomic OMP add_wrapped_subgrid
* Leslie Greengard and June-Yub Lee - CMCL Fortran test drivers
* Dan Foreman-Mackey - early python wrappers
* David Stein - python wrappers, finding "pi-1ULP" spreadcheck error
* Vineet Bansal - py packaging
* Garrett Wright - dual-precision build, py packaging, GPU version

Testing, bug reports, helpful discussions:

* Dan Fortunato - MATLAB setpts temp array bug and fix
* Hannah Lawrence - user testing and finding bugs
* Marina Spivak - Fortran testing
* Hugo Strand - python bugs
* Amit Moscovich - Mac OSX build
* Dylan Simon - sphinx help
* Zydrunas Gimbutas - MWrap extension, explanation that NFFT uses Kaiser-Bessel backwards
* Charlie Epstein - help with analysis of kernel Fourier transform sums
* Christian Muller - optimization (CMA-ES) for early kernel design
* Andras Pataki - complex number speed in C++, thread-safety of FFTW
* Jonas Krimmer - thread safety of FFTW, Windows makefile
* Timo Heister - pass/fail numdiff testing ideas
* Vladimir Rokhlin - piecewise polynomial approximation on complex boxes

Logo design: `Sherry Choi <http://www.sherrychoi.com>`_ with input
from Alex Barnett and Lucy Reading-Ikkanda.
  
We are also indebted to the authors of other NUFFT codes
such as NFFT3, CMCL NUFFT, MIRT, BART, etc, upon whose interfaces, code,
and algorithms we have built.
