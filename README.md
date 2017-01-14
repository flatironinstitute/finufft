# Flatiron Institute Nonuniform Fast Fourier Transform libraries: FINUFFT

### Magland, Greengard, Barnett

Includes code by:

P. Swarztrauber - FFTPACK
Tony Ottosson - evaluate modified Bessel function K_0
June-Yub Lee - some test codes co-written with Greengard


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
* compiler directives to link against FFTW, for non-spreading-dominated problems.
* cnufft->finufft names

### Done

* efficient modulo in spreader
* removed data-zeroing bug in t-II spreader
