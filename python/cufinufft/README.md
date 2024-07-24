# FINUFFT GPU library Python wrappers

This is a Python interface to the efficient GPU CUDA implementation of the 1-, 2- and
3-dimensional nonuniform fast Fourier transform (NUFFT), provided
in the FINUFFT library. It performs type
1 (nonuniform to uniform) or type 2 (uniform to nonuniform) transforms.
For a mathematical description of the NUFFT and applications to signal
processing, imaging, and scientific computing, see [the FINUFFT
documentation](https://finufft.readthedocs.io).
The Python GPU interface is [here](https://finufft.readthedocs.io/en/latest/python_gpu.html).
Usage examples can be found in the examples folder in the same directory as
the file you are reading.

If you use this GPU feature of our package, please cite our GPU paper:

Y. Shih, G. Wright, J. Andén, J. Blaschke, A. H. Barnett (2021).
cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs.
arXiv preprint arXiv:2102.08463.
[(paper)](https://arxiv.org/abs/2102.08463)
[(bibtex)](https://arxiv.org/bibtex/2102.08463)

**Note**: With version 2.2 we have changed the GPU interfaces slightly to better align with FINUFFT. For an outline of the changes, please see [the migration guide](https://finufft.readthedocs.io/en/latest/cufinufft_migration.html).
