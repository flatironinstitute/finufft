.. _ackn:

Acknowledgments
===============

FINUFFT was initiated by Jeremy Magland and Alex Barnett at the
Center for Computational Mathematics, Flatiron Institute (then called Simons Center for Data Analysis) in early 2017.
The main developer and maintainer is:

* Alex Barnett

Major code contributions (loosely in chronological order) by:

* Jeremy Magland - early multithreaded spreader, benchmark vs other libraries, py wrapper
* Ludvig af Klinteberg - SIMD vectorization/acceleration of spreader, Julia wrapper
* Yu-Hsuan ("Melody") Shih - 2d1many, 2d2many vectorized interface, GPU version, including 1D
* Andrea Malleo - guru interface prototype and tests
* Libin Lu - guru Fortran, python/wheels, MATLAB/MEX (including GPU), Julia interfaces, cmake, CI, user support, SIMD kernel, PSWF optimization and integration, detective work...
* Joakim Andén - python, MATLAB/FFTW issues, dual-precision, performance tests, GPU python/docs/tests
* Robert Blackwell - atomic OMP add_wrapped_subgrid, GPU version merge
* Marco Barbone - SIMD kernel manual vectorization, benchmarking, Cmake/packaging, Windows, CI, GPU type 3 and output-driven method, on-the-fly kernel polynomials...
* Martin Reinecke - early SIMD kernel and interp auto-vectorization, binsort, switchable FFT to DUCC0, exploiting FFT zero blocks, de-macro-izing, malloc reduction, adjoint execute, good ideas...

Other contributors to code either directly or indirectly include:

* Leslie Greengard and June-Yub Lee - CMCL Fortran test drivers
* Dan Foreman-Mackey - early python wrappers
* David Stein - python wrappers, finding "pi-1ULP" spreadcheck error
* Garrett Wright - dual-precision build, py packaging, GPU version
* Wenda Zhou - Cmake build, code review, professionalization, SIMD ideas
* Vladimir Rokhlin - PSWF evaluation (C versions of legeexps and prolcrea)

Testing, bug reports, helpful discussions, contributions:

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
* Vladimir Rokhlin - idea of piecewise polynomial approximation on complex boxes (used up to v2.4.1)
* Reinhard Neder - fortran90 demo using finufft as module, OSX build
* Vineet Bansal - py packaging
* Jason Kaye - Gauss-Legendre quadrature code from cppdlr
* Juan Ignacio Polanco - GPU output driven idea, discussions
* Julius Herb - Poisson equation tutorial in Python
* Felix F. Zimmermann - Python dependency issues in cufinufft
* Yuwei Sun - Available thread count fix
* Maxim Ermenko - CUDA type 3 simple interface in Python

Logo design: `Sherry Choi <http://www.sherrychoi.com>`_ with input
from Alex Barnett and Lucy Reading-Ikkanda.

We are also indebted to the authors of other NUFFT codes
such as NFFT3, CMCL NUFFT, MIRT, BART, DUCC0, NonuniformFFTs.jl, etc,
upon whose interfaces, code, and algorithms we have built.
