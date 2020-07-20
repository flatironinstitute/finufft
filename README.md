# cuFINUFFT
A GPU implementation of 2, 3 dimension type 1, 2 non-uniform FFT based on [FINUFFT][1].

This is a work from Melody Shih's internship at Flatiron Institute, advised by CCM project leader Alex Barnett.

## Installation

Note for most Python users, you may skip to the [Python Package](#Python-Package) section first,
and consider installing from source if that solution is not adequate for your needs.

 - Get this code -
 ```git clone https://github.com/flatironinstitute/cufinufft.git```
 - Review the `makefile`. - If you need to customize build settings, create and edit a `make.inc`.  Example:
   - To override the standard CUDA `/usr/local/cuda` location your `make.inc` should contain: ```CUDA_ROOT=/your/path/to/cuda```.
   - Two examples are provided, one for IBM machines (make.inc.power9), and another for the Courant Institute cluster (make.inc.CIMS).
 - Compile - ```make all -j```
 - Run test codes - ```make check```

### Library Installation

It is up to the user to decide how exactly to link or otherwise install the libraries produced in `lib`.
If you plan to use the Python wrapper you will minimally need to extend your `LD_LIBRARY_PATH`,
such as with `export LD_LIBRARY_PATH=${PWD}/lib:${LD_LIBRARY_PATH}` or a more permanent installation
path of your choosing.

If you would like to always have this installation in your library path, you can add to your shell rc
with something like the following:

`echo "\n# cufinufft librarypath \nexport LD_LIBRARY_PATH=${PWD}/lib:${LD_LIBRARY_PATH}" >> ~/.bashrc`

Because CUDA itself has similar library/path requirements, it is expected the user is somewhat familiar.
If not, please ask, we might be able to help.

### Python Wrapper

For those installing from source, this code comes with a Python wrapper module `cufinufftpy`.
Once you have successfully installed and tested the CUDA library
you may run `make python` to manually install the additional Python package.

### Python Package

General Python users, or Python software packages which would like to automatically
depend on cufinufftpy using `setuptools` may use a precompiled binary distribution.
This totally avoids installing from source and managing libraries for supported systems.

Because binary distributions are specific to both hardware and software,
we currently only support systems covered by `manylinux2010` that are using
CUDA 10.1, 10.2, or 11.0-rc with a compatible GPU. This is currently the most
common arrangement.  If you have such a system, you may run:

`pip install cufinufftpy`

For other cases, the Python wrapper should be able to be built from source.
We hope to extend this in the future, and have begun work for `manylinux2014`.
 
## Interface
cuFINUFFT API contains 5 stages:
 - Set cufinufft default options - ```int ier=cufinufft_default_opts(type1, dim, &dplan.opts);```
 - Make cufinufft plan - ``` ier=cufinufft_makeplan(type1, dim, nmodes, iflag, ntransf, tol, maxbatchsize, &dplan); ```
 - Set the locations of non-uniform points x,y,z - ```ier=cufinufft_setpts(M, x, y, z, 0, NULL, NULL, NULL, &dplan);```
 - Apply the transformation with data c,fk - ```ier=cufinufft_exec(c, fk, &dplan); ```
 - Destroy cufinufft plan - ```ier=cufinufft_destroy(&dplan);```
 
## Preprocessors
 - TIME - timing for each stage.  Enable by adding "-DTIME" to `NVCCFLAGS`.
 - SPREADTIME - more detailed timing from spreading and interpolation
 - DEBUG - debug mode outputs all the middle stages' result
 
## Other
 - If you are interested in optimizing for GPU Compute Capability,
 you may want to specicfy ```NVARCH=-arch=sm_XX``` in your make.inc to reduce compile times,
 or for other performance reasons. See [Matching SM Architectures][2].

[1]: https://github.com/flatironinstitute/finufft
[2]: http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
