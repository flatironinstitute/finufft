# Flatiron Institute Nonuniform Fast Fourier Transform libraries: FINUFFT

**Alex H. Barnett and Jeremy F. Magland**, with additional code by
Ludvig af Klinteberg and Yu-hsuan "Melody" Shih.

<p>
<img src="docs/logo.png" width="350"/>
<img src="docs/spreadpic.png" width="400"/>
</p>

This is a lightweight library to compute the three standard types of nonuniform FFT to a specified precision, in one, two, or three dimensions. It is written in C++ with interfaces to C, Fortran, MATLAB/octave, and python. A julia interface
also exists.

Please see the [online documentation](http://finufft.readthedocs.io/en/latest/index.html), or its equivalent, the [user manual](finufft-manual.pdf).
You will also want to see example codes in the directories
`examples`, `test`, `fortran`, `matlab`, and `python_tests`.
If you cannot compile, try our [precompiled binaries](http://users.flatironinstitute.org/~ahb/codes/finufft-binaries).

If you prefer to read text files, the source to generate the above documentation is in human-readable (mostly .rst) files as follows:

- `docs/install.rst` : installation and compilation instructions
- `docs/math.rst` : mathematical definitions
- `docs/dirs.rst` : explanation of directories and files in the package
- `docs/usage.rst` : interfaces from C++, C and Fortran, notes on parameters
- `docs/matlabhelp.raw` : MATLAB/octave interfaces
- `finufftpy/_interfaces.py` : python interface docstrings
- `docs/related.rst` : packages using FINUFFT
- `docs/issues.rst` : known issues and bug reports
- `docs/refs.rst` : journal article references
- `docs/ackn.rst` : acknowledgments

If you find FINUFFT useful in your work, please cite this code and
our paper:

A parallel non-uniform fast Fourier transform library based on an ``exponential of semicircle'' kernel.
A. H. Barnett, J. F. Magland, and L. af Klinteberg.
SIAM J. Sci. Comput. 41(5), C479-C504 (2019).
