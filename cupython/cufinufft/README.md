# cuFINUFFT v1.2 Python package

The cuFINUFFT library is an efficient GPU implementation of the 1-, 2- and
3-dimensional nonuniform fast Fourier transform (NUFFT). It includes both type
1 (nonuniform to uniform) and type 2 (uniform to nonuniform) transforms.
It is based on the [FINUFFT](https://github.com/flatironinstitute/finufft)
implementation for the CPU. This package provides a Python interface to the
cuFINUFFT library, which is written in C++ and CUDA.

For a mathematical description of the NUFFT and applications to signal
processing, imaging, and scientific computing, see [the FINUFFT
documentation](https://finufft.readthedocs.io). Usage examples can be found
[here](https://github.com/flatironinstitute/cufinufft/tree/v1.2/examples).

If you use this package, please cite our paper:

Y. Shih, G. Wright, J. Andén, J. Blaschke, A. H. Barnett (2021).
cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs.
arXiv preprint arXiv:2102.08463.
[(paper)](https://arxiv.org/abs/2102.08463)
[(bibtex)](https://arxiv.org/bibtex/2102.08463)

**Note**: We are currently in the process of adapting the cuFINUFFT interface to
closer match that of FINUFFT. This will likely break code depending on the
current interface once the next release is published. At this point we will
publish a migration guide that will detail the exact changes to the
interfaces.
