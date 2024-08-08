.. _devnotes:

Developer notes
===============

* Developers needing to update/regenerate documentation in general, including our readthedocs website, see ``docs/README``. Developers changing MATLAB/octave interfaces or docs, also see ``matlab/README``. Please also see ``contributing.md`` for code style and git hook guidelines.

* To update the version number, this needs to be done by hand in the following places:

  - ``CMakeLists.txt`` for cmake
  - ``docs/conf.py`` for sphinx
  - ``docs/install.rst`` cmake git tags
  - ``python/finufft/finufft/__init__.py`` for the python pkg version
  - ``python/cufinufft/cufinufft/__init__.py`` for the GPU python pkg version
  - ``include/finufft/defs.h`` for the debug>0 output
  - ``matlab/Contents.m`` for the MATLAB/Octave help
  - ``CHANGELOG``: don't forget to describe the new features and changes, folding lines at 80 chars.

* There are some sphinx tags in the source code, indicated by @ in comments. Please leave these alone since they are needed by the doc generation.

* Source code is now in clang format: devs should run ``clang-format --files=<editedfile> -i --style=.clang-format`` before pushing, or set up their editor to do this
  automatically.

* If you add a new option field (recall it must be plain C style only, no special types) to ``include/finufft_opts.h``, don't forget to add it to ``include/finufft.fh``, ``include/finufft_mod.f90``, ``matlab/finufft.mw``, ``python/finufft/_finufft.py``, and the Julia interface, as well a paragraph describing its use in the docs. Also to set its default value in ``src/finufft.cpp``. You will then need to regenerate the docs as in ``docs/README``.

* For testing and performance measuring routines see ``test/README`` and ``perftest/README``. We need more of the latter, eg, something making performance graphs that enable rapid eyeball comparison of various settings/machines. Marco is working on that.

* The kernel function in spreadinterp is evaluated via piecewise-polynomial approximation (Horner's rule). The code for this is auto-generated in MATLAB, for all upsampling factors. There are two versions supported:

  - 2018--2024 vintage: no explicit SIMD vectorization, C code is generated code for the Horner evaluation loop, by running from MATLAB ``gen_all_horner_C_code.m``

  - post-2024 vintage: explicit SIMD and many other acceleration tricks, and the generated code is a static C++ array of coefficients, and their sizes (``nc`` or number of coefficients) for each width ``w``. Run from MATLAB ``gen_ker_horner_loop_cpp_code.m``

  See ``devel/README`` for more details. The ES kernel coefficient and poly approx degree for both of the above are defined in a single location, ``devel/get_degree_and_beta.m``, which must match the C++ ``setup_spreader()`` function.

* Continuous Integration (CI). See files for this in ``.github/workflows/``. It currently tests the default ``makefile`` settings in linux, and three other ``make.inc.*`` files covering OSX and Windows (MinGW). CI does not test build the variant OMP=OFF. The dev should test these locally. Likewise, the Julia wrapper is separate and thus not tested in CI. We have added ``JenkinsFile`` for the GPU CI via python wrappers.

* **Installing MWrap**. This is needed only for experts to rebuild the matlab/octave interfaces.
  `MWrap <https://github.com/zgimbutas/mwrap>`_
  is a very useful MEX interface generator by Dave Bindel, now maintained
  and expanded by Zydrunas Gimbutas.
  Make sure you have ``flex`` and ``bison`` installed to build it.
  As of FINUFFT v2.0 you will need a recent (>=0.33.10) version of MWrap.
  Make sure to override the location of MWrap by adding a line such as::

    MWRAP = your-path-to-mwrap-executable

  to your ``make.inc``, and then you can use the ``make mex`` task.

* The cufinufft Python wheels are generated using Docker based on the manylinux2014 image. For instructions, see ``tools/cufinufft/distribution_helper.sh``. These are binary wheels that are built using CUDA 11 (or optionally CUDA 12, but these are not distributed on PyPI) and bundled with the necessary libraries.

* CMake compiling on linux at Flatiron Institute (Rusty cluster): We have had a report that if you want to use LLVM, you need to ``module load llvm/16.0.3`` otherwise the default ``llvm/14.0.6`` does not find ``OpenMP_CXX``.

* Testing cufinufft (for FI, mostly):

.. code-block:: sh

    # to grab an interactive GPU shell -- here with 10 cores for building and a v100 for
    # testing. You could just as easily try this on your workstation
    srun -p gpu -C v100 -c 10 -n 1 --gpus=1 --pty bash
    cd path/to/finufft

    # get the local card to this machine's compute capability. If you know it you can obviously type it yourself
    CUDAARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader|head -n 1| sed 's/\.//')

    # Load modules and build a venv. We typically recommend using venvs that fall back our our
    # default packages with the python module (--system-site-packages)
    module -q purge
    module -q load gcc python cmake fftw cuda
    python -m venv venv --system-site-packages
    source venv/bin/activate
    pip install --upgrade pip

    # building. Feel free to tweak whatever
    mkdir -p build && cd build
    cmake -DFINUFFT_BUILD_TESTS=on -DFINUFFT_BUILD_EXAMPLES=on -DFINUFFT_USE_CUDA=on \
        -DCMAKE_CUDA_ARCHITECTURES=$CUDAARCH -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
    make -j 10

    # for standard tests
    ctest

    # python install. Needs build from before since installer searches for libcufinufft.so in
    # LD_LIBRARY_PATH (and default path)
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD pip install -e ../python/cufinufft

    # python tests. we have other GPU framework support, but you need to make sure they're
    # installed (numba, pycuda, torch, cupy). This LD_LIBRARY_PATH may or may not be necessary,
    # depending on if an RPATHing issue appears. Fix upstream at time of writing
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD python -m pytest --framework=numba ../python/cufinufft
