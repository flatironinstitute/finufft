.. _install_gpu:

Installation (GPU)
==================

.. note::

    Python users may install the cuFINUFFT package using ``pip install cufinufft``, which contains binary wheels compiled against CUDA 11.2 on Linux. If these requirements do not work for your use case, please see the detailed instructions below.

The GPU version of FINUFFT is called cuFINUFFT,
and it uses CUDA kernels (often exploiting fast GPU shared memory)
to speed up spreading/interpolation operations, as well as cuFFT.
It operates on GPU arrays, which enables low-overhead integration with other GPU processing pipelines, but does requires the user to transfer their data between the host (CPU) and the device (GPU).
See the main :ref:`overview page<index>` and :ref:`reference<refs>` [S21] for more details.
It is currently being tested on the Linux platform, but you should be able to adapt the instructions below to work on other platforms, such as Windows and macOS.

CMake installation
------------------

To automate the installation process, we use ``cmake``. To use this, run

.. code-block:: bash

    mkdir build
    cd build
    cmake -D FINUFFT_USE_CUDA=ON ..
    cmake --build .

The ``libcufinufft.so`` (along with ``libfinufft.so``) will now be present in your ``build`` directory. Note that for this to work, you must have the Nvidia CUDA toolchain installed (such as the ``nvcc`` compiler, among others). To speed up the compilation, you could replace the last command by ``cmake --build . -j`` to use all threads,
or ``cmake --build . -j8`` to specify using 8 threads, for example.
To avoid building the CPU library (``libfinufft.so``), you can set the ``FINUFFT_USE_CPU`` flag to ``OFF``.

In order to configure cuFINUFFT for a specific compute capability, use the ``CMAKE_CUDA_ARCHITECTURES`` flag. For example, to compile for compute capability 8.0 (supported by Nvidia A100), replace the 3rd command above by

.. code-block:: bash

    cmake -D FINUFFT_USE_CUDA=ON -D CMAKE_CUDA_ARCHITECTURES=80 ..

To find out your own device's compute capability without having to look it up on the web, use:

.. code-block:: bash

    nvidia-smi --query-gpu=compute_cap --format=csv,noheader

This will return a text string such as ``8.6`` which would incidate
``sm_86`` architecture, thus to use ``CMAKE_CUDA_ARCHITECTURES=86``.


Note that by default the ``CMAKE_CUDA_ARCHITECTURES`` flag is set to ``native``, which means that the code will be compiled for the compute capability of the GPU on which the code is being compiled.
This might not be portable so it is recommended to set this flag explicitly when building for multiple systems. A good alternative is ``all-major`` which will compile for all major compute capabilities.


Testing
-------

To test your cuFINUFFT package, configure it with the ``BUILD_TESTING`` and ``FINUFFT_BUILD_TESTS`` flags set to ``ON``. In other words, run

.. code-block:: bash

    cmake -D FINUFFT_USE_CUDA=ON -D BUILD_TESTING=ON -D FINUFFT_BUILD_TESTS=ON ..

Then after compiling as above with ``cmake --build .``, you execute the tests using

.. code-block:: bash

    cmake --build . -t test

This runs a suite of GPU accuracy (mathematical correctness) and interface API tests. See the ``test/cuda/`` directory for individual usage and documentation of these tests.


Python interface
----------------

.. _install-python-gpu:

In addition to the C interface, cuFINUFFT also comes with a Python interface. As mentioned above, this can be most easily installed by running ``pip install cufinufft``, but it can also be installed from source. The Python interface code is located in the ``python/cufinufft`` subdirectory, so to install it, you first build the shared library as seen above, then run

.. code-block:: bash

    pip install python/cufinufft

Note that since cuFINUFFT supports a number of different GPU frameworks (CuPy, Numba, PyTorch, and PyCuda), it does not install any of these automatically as a dependency.
You must therefore install one of these manually.
For example, for CuPy, you would run

.. code-block:: bash

    pip install cupy-cuda11x

for the CUDA 11.2--11.x version of CuPy.
Assuming ``pytest`` is installed (otherwise, just run ``pip install pytest``), you can now test the installation by running

.. code-block:: bash

    pytest --framework=cupy python/cufinufft/tests

In contrast to the C interface tests, these check for correctness, so a successful test run signifies that the library is working correctly.
Note that you can specify other framework (``pycuda``, ``torch``, or ``numba``) for testing using the ``--framework`` argument.


Matlab interface
----------------

.. _install-matlab-gpu:

As of version 2.4, cuFINUFFT also comes with a MATLAB GPU ``gpuArray`` interface. To install this, you first build the shared library.
For example, assuming in the root directory of FINUFFT, then run

.. code-block:: bash

    cmake -S . -B build -D FINUFFT_USE_CUDA=ON -D FINUFFT_STATIC_LINKING=OFF -D CMAKE_VERBOSE_MAKEFILE:BOOL=ON -D CMAKE_CUDA_ARCHITECTURES=native

You may adjust ``CMAKE_CUDA_ARCHITECTURES`` to generate the code for different compute capabilities, e.g., use ``all-major`` which will compile for all major compute capabilities.
Then build the binary library

.. code-block:: bash

    cmake --build build

Then, to compile (on Linux or OSX platforms, at least) the MATLAB mexcuda executable, open MATLAB in the FINUFFT root directory and run

.. code-block:: matlab

    mexcuda -v 'LINKLIBS=$LINKLIBS -Wl,-rpath,/absolute/path/to/finufft/build -Lbuild -lcufinufft' matlab/cufinufft.cu -Iinclude -DR2008OO -largeArrayDims -output matlab/cufinufft

``-Lbuild`` specifies the relative path where ``libcufinufft.so`` is placed during the linking stage. ``-Wl,-rpath,/absolute/path/to/finufft/build`` specifies the absolute path where ``libcufinufft.so`` is, so that MATLAB can find it during runtime; change ``/absolute/path/to/finufft/build`` accordingly. You may remove ``-Wl,-rpath,/absolute/path/to/finufft/build``, you then need to export `LD_LIBRARY_PATH` to include path to `libcufinufft.so` so that MATLAB can find it during runtime.

You should now test your installation by opening MATLAB, then
``addpath matlab`` then ``run matlab/test/fullmathtest``, which should
complete CPU (if present) and GPU tests in a couple of seconds.

.. note::

    Depending on your MATLAB version, ``mexcuda`` compiles the CUDA code using the NVIDIA ``nvcc`` compiler installed with MATLAB. If the MATLAB default one does not work, you may specify the location of ``nvcc`` on your system by storing it in the environment variable ``MW_NVCC_PATH``, eg via ``setenv("MW_NVCC_PATH","/path/to/CUDA/bin")`` and ``setenv("MW_ALLOW_ANY_CUDA","true")``. You may also check `toolbox/parallel/gpu/extern/src/mex/glnxa64/nvcc_g++.xml` to see how MATLAB finds the ``nvcc`` compiler.

.. note::

   We do not have a ``makefile`` task for building the MATLAB GPU interface, since ``libcufinufft.so`` is built in CMake instead of the makefile. A CMake mexcuda task for the above, and the Windows commands, are on the to-do list.
