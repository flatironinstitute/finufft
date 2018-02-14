Acknowledgments
===============

The main code and mathematical development is by:

* Alex Barnett (Flatiron Institute)
* Jeremy Magland (Flatiron Institute)
    
Significant SIMD/vectorization acceleration of the spreader is by:

* Ludvig af Klinteberg (SFU)

Other code contributions:

* Leslie Greengard and June-Yub Lee - CMCL fortran drivers and test codes
* Dan Foreman-Mackey - python wrappers
* David Stein - python wrappers
* Dylan Simon - sphinx help
  
Testing, bug reports:

* Joakim Anden - catching memory leak
* Hannah Lawrence - user testing and finding bugs
* Marina Spivak - fortran testing
* Hugo Strand - python bugs
  
Helpful discussions:

* Charlie Epstein - analysis of kernel Fourier transform sums
* Christian Muller - optimization (CMA-ES) for early kernel design
* Andras Pataki - complex number speed in C++
* Timo Heister - pass/fail numdiff testing ideas
* Zydrunas Gimbutas - explanation that NFFT uses Kaiser-Bessel backwards
