Python interface (GPU)
======================

Quick-start examples
--------------------

As mentioned in the :ref:`Python GPU installation instructions <install-python-gpu>`, the easiest way to install the Python interface for cuFINUFFT is to run ``pip install cufinufft``.

Assuming cuFINUFFT has been installed, we will now consider how to calculate a 1D type 1 transform.
To manage the GPU and transfer to and from host and device, we will use the ``pycuda`` library.
Consequently, we start with a few import statements.

.. code-block:: python

    import numpy as np

    import pycuda.autoinit
    from pycuda.gpuarray import GPUArray, to_gpu

    from cufinufft import cufinufft

We then proceed to setting up a few parameters.

.. code-block:: python

    # number of nonuniform points
    M = 100000

    # grid size
    N = 200000

    # generate positions for the nonuniform points and the coefficients
    x = 2 * np.pi * np.random.uniform(size=M)
    c = (np.random.standard_normal(size=M)
         + 1J * np.random.standard_normal(size=M))

Now that the data is prepared, we need to set up a cuFINUFFT plan that can be executed on that data.

.. code-block:: python

    # create plan
    plan = cufinufft(1, (N,), dtype=np.float64)

    # set the nonuniform points
    plan.set_pts(to_gpu(x))

The cuFINUFFT interface relies on the user to supply a preallocated output array, so we use ``pycuda.GPUArray`` for this:

.. code-block:: python

    # allocate output array
    f_gpu = GPUArray((N,), dtype=np.complex128)

With everything set up, we are now ready to execute the plan.

.. code-block:: python

    # execute the plan
    plan.execute(to_gpu(c), f_gpu)

    # move results off the GPU
    f = f_gpu.get()

The last line is needed here to copy the data from the device (GPU) to the host (CPU).
Now ``f`` is a size-``N`` array containing the NUDFT of ``c`` at the points ``x``.

Other possible calculations are possible by supplying different options during plan creation.
See the full API documentation below for more information.

Full documentation
------------------

.. automodule:: cufinufft
    :members:
    :member-order: bysource
