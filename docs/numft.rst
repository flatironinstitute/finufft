.. _numft:

Numerical computation of continuous Fourier transforms of functions
===================================================================

It is common to assume that the FFT is the right tool to compute
*continuous Fourier transforms*, but this is not so, unless you are
content with very poor accuracy.
The reason is that the FFT applies to equispaced data samples,
that is, a quadrature scheme with only equispaced nodes.




