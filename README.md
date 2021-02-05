# cuFINUFFT v1.1

<img align="right" src="docs/logo.png" width="350">

cuFINUFFT is a very efficient GPU implementation of the 2- and 3-dimensional non-uniform FFT of types 1 and 2, in single and double precision, based on the CPU code [FINUFFT][1].
It has several algorithmic innovations, including load-balancing, bin-sorting
for cache-aware access, and use of fast shared memory.
Our tests show an acceleration above FINUFFT of up to 10x on modern hardware,
and up to 100x faster than some other established GPU NUFFT codes.
The transforms it performs may be summarized as follows: type 1 maps nonuniform data to a bi- or tri-variate Fourier series,
whereas type 2 does the adjoint operation (which is not generally the inverse of type 1).
See the [documentation for FINUFFT][3] for a full mathematical description of the transforms and their applications to signal processing, imaging, and scientific computing.

Main developer: **Yu-hsuan Melody Shih** (NYU). Main other contributors:
Garrett Wright (Princeton), Joakim Andén (KTH/Flatiron), Johannes Blashcke (LBNL), Alex Barnett (Flatiron).
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


## Usage and interface

Please see the codes in `examples/` to see how to call cuFINUFFT
and link to from C++, and to call from Python.

The standard use of the cuFINUFFT API has four stages:
 - Planning one or more transforms - ```ier=cufinufft_makeplan(type1, dim, nmodes, iflag, ntransf, tol, maxbatchsize, &dplan, NULL); ```
 - Set the locations of non-uniform points `x`, `y`, and possibly `z` - ```ier=cufinufft_setpts(M, x, y, z, 0, NULL, NULL, NULL, dplan);```
 - Apply the transformation (input `c` to output `fk` for type 1, or vice versa for type 2) - ```ier=cufinufft_execute(c, fk, dplan); ```
 - Destroy the plan (clean up) - ```ier=cufinufft_destroy(dplan);```

It is also possible to change advanced options by changing the last `NULL`
argument of the above `cufinufft_makeplan` call to a pointer
to an options struct, `opts`.
This struct should first be initialized via
```cufinufft_default_opts(type, dim, &opts);```
For examples of such advanced usage, see `test/cufinufft*.cu`.


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

Because binary distributions are specific to both hardware and software,
we currently only support systems covered by `manylinux2010` that are using
CUDA 10.1, 10.2, or 11.0-rc with a compatible GPU. This is currently the most
common arrangement.  If you have such a system, you may run:

`pip install cufinufft`

For other cases, the Python wrapper should be able to be built from source.
We hope to extend this in the future, and have begun work for `manylinux2014`.
 
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

[1]: https://github.com/flatironinstitute/finufft
[2]: http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
[3]: https://finufft.readthedocs.io