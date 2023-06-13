.. _install_gpu:

Installation (GPU)
==================

.. note::

    Python users may install the cuFINUFFT package using ``pip install cufinufft``, which contains binary wheels compiled against CUDA 10.2 on Linux. If these requirements do not work for your use case, please see the detailed instructions below.

The GPU version of FINUFFT is called cuFINUFFT,
and it uses CUDA kernels (often exploiting fast GPU shared memory)
to speed up spreading/interpolation operations, as well as cuFFT.
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

In order to configure cuFINUFFT for a specific compute capability, use the ``CMAKE_CUDA_ARCHITECTURES`` flag. For example, to compile for compute capability 8.0 (supported by NVidia A100), replace the 3rd command above by

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

Note that these tests only checks if the compiled GPU code executes – it currently does not verify accuracy (i.e., whether the code executes *correctly*).
This will be remedied shortly.


Python interface
----------------

.. _install-python-gpu:

In addition to the C interface, cuFINUFFT also comes with a Python interface. As mentioned above, this can be most easily installed by running ``pip install cufinufft``, but it can also be installed from source. The Python interface code is located in the ``cupython`` subdirectory, so to install it, you run

.. code-block:: bash

    cd cupython
    LD_LIBRARY_PATH="../build" LIBRARY_PATH="../build" pip install .

Note that the ``LD_LIBRARY_PATH`` and ``LIBRARY_PATH`` environment variables must be set for the Python interpreter to find ``libcufinufft.so`` (assuming it has not been installed in the appropriate system directory).

Assuming ``pytest`` is installed (otherwise, just run ``pip install pytest``), you can now test the installation by running

.. code-block:: bash

    LD_LIBRARY_PATH="../build" pytest

Again, ``LD_LIBRARY_PATH`` must be set in order for the interpreter to find the shared library. This applies to any invocation of the Python interpreter when using the ``cufinufft`` package. (``LIBRARY_PATH`` is no longer necessary since there is no compilation at this stage.) In contrast to the C interface tests, these check for correctness, so a successful test run signifies that the library is working correctly.
