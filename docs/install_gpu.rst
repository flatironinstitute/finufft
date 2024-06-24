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
