# Flatiron Institute Nonuniform Fast Fourier Transform libraries: FINUFFT

### Alex H. Barnett and Jeremy F. Magland

<img src="docs/logo.png" width="350"/>
<img src="docs/spreadpic.png" width="400"/>

This is a lightweight library to compute the three standard types of nonuniform FFT to a specified precision, in one, two, or three dimensions. It is written in C++ with interfaces to C, Fortran, MATLAB/octave, and python. A julia interface
also exists.

Please see the [online documentation](http://finufft.readthedocs.io/en/latest/index.html), or its equivalent, the [user manual](finufft-manual.pdf).
You will also want to see example codes in the directories
`examples`, `test`, `fortran`, `matlab`, and `python_tests`.

If you prefer to read text files, the source to generate the above documentation is in human-readable (mostly .rst) files as follows:

- `docs/install.rst` : installation instructions
- `docs/math.rst` : mathematical definitions
- `docs/dirs.rst` : explanation of directories and files in the package
- `docs/usage.rst` : C++ routine interfaces, compilation options, and notes
- `docs/matlabhelp.raw` : MATLAB/octave interfaces
- `finufftpy/_interfaces.py` : python interface docstrings
- `docs/issues.rst` : known issues and bug reports
- `docs/refs.rst` : journal article references
- `docs/ackn.rst` : acknowledgments

If you find FINUFFT useful in your work, please cite this code and
the forthcoming paper (see references).
