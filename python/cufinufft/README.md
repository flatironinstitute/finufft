# cuFINUFFT v1.2 Python package

The cuFINUFFT library is an efficient GPU implementation of the 2- and
3-dimensional nonuniform fast Fourier transform (NUFFT). It includes both type
1 (nonuniform to uniform) and type 2 (uniform to nonuniform) transforms.
It is based on the [FINUFFT](https://github.com/flatironinstitute/finufft)
implementation for the CPU. This package provides a Python interface to the
cuFINUFFT library, which is written in C++ and CUDA.

For a mathematical description of the NUFFT and applications to signal
processing, imaging, and scientific computing, see [the FINUFFT
documentation](https://finufft.readthedocs.io).

Usage examples can be found
[here](https://github.com/flatironinstitute/cufinufft/tree/v1.2/examples).
