# Flatiron Institute Nonuniform Fast Fourier Transform libraries: FINUFFT

### Magland, Greengard, Barnett

Includes code by:

P. Swarztrauber - FFTPACK
Tony Ottosson - evaluate modified Bessel function K_0
June-Yub Lee - some test codes co-written with Greengard

### Purpose

to do

### Dependencies

Fortran 77 compiler, C++ compiler, GNU make.
To run optional speed comparisons which link against the CMCL NUFFT library, this must be installed.

### Installation

1. Download using `git`, `svn`, or as a zip (see green button above).
1. `cp makefile.dist makefile`
1. edit `makefile` for your system
1. `make`


### Contents of this package

  `src` : source code and headers for libraries (mixture of Fortran 77 and C++).
  `examples` : test codes (drivers) which verify libaries are working correctly, and show how to call them.
  `contrib` : 3rd-party code.
  `matlab` : wrappers and examples for MATLAB.
  `doc` : the manual.
  `README.md`
  `LICENSE`
  `makefile.dist` : GNU makefile (copy to `makefile` first)
  

### References

This code builds upon the CMCL NUFFT, for which the following are references:

[1] Accelerating the Nonuniform Fast Fourier Transform: (L. Greengard and J.-Y. Lee) SIAM Review 46, 443 (2004).
[2] The type 3 nonuniform FFT and its applications: (J.-Y. Lee and L. Greengard) J. Comput. Phys. 206, 1 (2005).

For the original NUFFT paper, see

Fast Fourier Transforms for Nonequispaced data: (A. Dutt and V. Rokhlin) SIAM J. Sci. Comput. 14, 1368 (1993). 

### To do

* t-1 matlab and fortran tests checking grid total
* t-II spreader fortran and matlab tests
* build universal index mappers
* t-I, t-II convergence params test: M/N and KB params
* overall scale factor understand in KB
* openMP spread in the 1 and 2 direction
* check J's bessel10 approx is ok.
* make ier report accuracy out of range, malloc size errors, etc
* spread_f needs ier output
* license file
* makefile opts and compiler directives to link against FFTW, for non-spreading-dominated problems.
* cnufft->finufft names
* alert Dan Foreman-Mackey re https://github.com/dfm/python-nufft
* doc/manual
* boilerplate stuff as in CMCL page

### Done

* efficient modulo in spreader
* removed data-zeroing bug in t-II spreader
* clean dir tree
