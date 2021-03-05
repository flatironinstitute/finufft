# cuFINUFFT v1.2

<img align="right" src="docs/logo.png" width="350">

cuFINUFFT is a very efficient GPU implementation of the 2- and 3-dimensional nonuniform FFT of types 1 and 2, in single and double precision, based on the CPU code [FINUFFT][1].

Note that the Python interface has changed relative to v1.1. Please see [CHANGELOG](CHANGELOG) for details.

cuFINUFFT introduces several algorithmic innovations, including load-balancing, bin-sorting for cache-aware access, and use of fast shared memory.
Our tests show an acceleration over FINUFFT of up to 10x on modern hardware,
and up to 100x faster than other established GPU NUFFT codes:

<p align="center">
<img src="docs/cufinufft_announce.png" width="550">
</p>

The transforms it performs may be summarized as follows: type 1 maps nonuniform data to a bi- or tri-variate Fourier series,
whereas type 2 does the adjoint operation (which is not generally the inverse of type 1).
These transforms are performed to a user-presribed tolerance,
at close-to-FFT speeds;
under the hood, this involves detailed kernel design, custom spreading/interpolation stages, and plain FFTs performed by cuFFT.
See the [documentation for FINUFFT][3] for a full mathematical description of the transforms and their applications to signal processing, imaging, and scientific computing.

Main developer: **Yu-hsuan Melody Shih** (NYU). Main other contributors:
Garrett Wright (Princeton), Joakim Andén (KTH/Flatiron), Johannes Blaschke (LBNL), Alex Barnett (Flatiron).
See github for full list of contributors.
This project came out of Melody's 2018 and 2019 summer internships at the Flatiron Institute, advised by CCM project leader Alex Barnett.

## Installation

Note for most Python users, you may skip to the [Python Package](#Python-Package) section first,
and consider installing from source if that solution is not adequate for your needs. Here's the C++ install process:

 - Make sure you have the prerequisites: a C++ compiler (eg `g++`) and a recent CUDA installation (`nvcc`).
 - Get the code: `git clone https://github.com/flatironinstitute/cufinufft.git`
 - Review the `Makefile`: If you need to customize build settings, create and edit a `make.inc`.  Example:
   - To override the standard CUDA `/usr/local/cuda` location your `make.inc` should contain: `CUDA_ROOT=/your/path/to/cuda`.
   - For examples, see one for IBM machines (`targets/make.inc.power9`), and another for the Courant Institute cluster (`sites/make.inc.CIMS`).
 - Compile: `make all -j` (this takes several minutes)
 - Run test codes: `make check` which should complete in less than a minute without error.
 - You may then want to try individual test drivers, such as `bin/cufinufft2d1_test_32 2 1e3 1e3 1e7 1e-3` which tests the single-precision 2D type 1. Most such executables document their usage when called with no arguments.


## Basic usage and interface

Please see the codes in `examples/` to see how to call cuFINUFFT
and link to from C++/CUDA, and to call from Python.

The default use of the cuFINUFFT API has four stages, that match
those of the plan interface to FINUFFT (in turn modeled on those of,
eg, FFTW or NFFT). Here they are from C++:
1. Plan one transform, or a set of transforms sharing nonuniform points, specifying overall dimension, numbers of Fourier modes, etc:

    ```c++
    ier = cufinufft_makeplan(type, dim, nmodes, iflag, ntransf, tol, maxbatchsize, &plan, NULL);
    ```

1. Set the locations of nonuniform points from the arrays `x`, `y`, and possibly `z`: 

    ```c++
    ier = cufinufft_setpts(M, x, y, z, 0, NULL, NULL, NULL, plan);
    ```

   (Note that here arguments 5-8 are reserved for future type 3 implementation, to match the FINUFFT interface).
1. Perform the transform(s) using these nonuniform point arrays, which reads strengths `c` and writes into modes `fk` for type 1, or vice versa for type 2:

    ```c++
    ier = cufinufft_execute(c, fk, plan);
    ```

1. Destroy the plan (clean up):

    ```c++
    ier = cufinufft_destroy(plan);
    ```

In each case the returned integer `ier` is a status indicator.
Here is [the full C++ documentation](docs/cppdoc.md).

It is also possible to change advanced options by changing the last `NULL`
argument of the `cufinufft_makeplan` call to a pointer
to an options struct, `opts`.
This struct should first be initialized via
```cufinufft_default_opts(type, dim, &opts);```
before the user changes any fields.
For examples of this advanced usage, see `test/cufinufft*.cu`


## Library installation

It is up to the user to decide how exactly to link or otherwise install the libraries produced in `lib`.
If you plan to use the Python wrapper you will minimally need to extend your `LD_LIBRARY_PATH`,
such as with `export LD_LIBRARY_PATH=${PWD}/lib:${LD_LIBRARY_PATH}` or a more permanent installation
path of your choosing.

If you would like to always have this installation in your library path, you can add to your shell rc
with something like the following:

`echo "\n# cufinufft librarypath \nexport LD_LIBRARY_PATH=${PWD}/lib:${LD_LIBRARY_PATH}" >> ~/.bashrc`

Because CUDA itself has similar library/path requirements, it is expected the user is somewhat familiar.
If not, please ask, we might be able to help.


## Python wrapper

For those installing from source, this code comes with a Python wrapper module `cufinufft`, which depends on `pycuda`.
Once you have successfully installed and tested the CUDA library,
you may run `make python` to manually install the additional Python package.

### Python package

General Python users, or Python software packages which would like to automatically
depend on cufinufft using `setuptools` may use a precompiled binary distribution.
This totally avoids installing from source and managing libraries for supported systems.

Binary distributions are specific to both hardware and software. We currently provide binary wheels targeting Linux systems covered by `manylinux2010` for CUDA 10 forward with compatible GPUs.  If you have such a system, you may run:

`pip install cufinufft`

For other cases, the Python wrapper should be able to be built from source.
 
## Advanced topics

### Advanced Makefile Usage

If you want to test/benchmark the spreader and interpolator
(the performance-critical components of the NUFFT algorithm),
without building the whole library, do this with `make checkspread`.

In general for make tasks,
it's possible to specify the target architecture using the `target` variable, eg:
```
make target=power9 -j
```
By default, the makefile assumes the `x86_64` architecture. We've included
site-specific configurations -- such as Cori at NERSC, or Summit at OLCF --
which can be accessed using the `site` variable, eg:
```
make site=olcf_summit
```

The currently supported targets and sites are:
1. Sites
    1. NERSC Cori (`site=nersc_cori`)
    2. NERSC Cori GPU (`site=nersc_cgpu`)
    3. OLCF Summit (`site=olcf_summit`) -- automatically sets `target=power9`
    4. CIMS (`target=CIMS`)
2. Targets
    1. Default (`x86_64`) -- do not specify `target` variable
    2. IBM `power9` (`target=power9`)

A general note about expanding the platform support: _targets_ should contain
settings that are specific to a compiler/hardware architecture, whereas _sites_
should contain settings that are specific to a HPC facility's software
environment. The `site`-specific script is loaded __before__ the
`target`-specific settings, hence it is possible to specify a target in a site
`make.inc.*` (but not the other way around).


### Makefile preprocessors
 - TIME - timing for each stage.  Enable by adding "-DTIME" to `NVCCFLAGS`.
 - SPREADTIME - more detailed timing from spreading and interpolation
 - DEBUG - debug mode outputs all the middle stages' result
 
### Other notes
 - If you are interested in optimizing for GPU Compute Capability,
 you may want to specicfy ```NVARCH=-arch=sm_XX``` in your make.inc to reduce compile times,
 or for other performance reasons. See [Matching SM Architectures][2].

## Tasks for developers

- We could use some help to implement 1D versions, and type 3 transforms (which are quite tricky), as in [FINUFFT][1]
- We need some more tutorial examples in C++ and Python
- Please help us to write MATLAB (gpuArray) and Julia interfaces
- Please see Issues for other things you can help fix


## References

* cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs,
Yu-hsuan Shih, Garrett Wright, Joakim Andén, Johannes Blaschke, Alex H. Barnett,
*accepted*, PDSEC2021. https://arxiv.org/abs/2102.08463


[1]: https://github.com/flatironinstitute/finufft
[2]: http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
[3]: https://finufft.readthedocs.io

