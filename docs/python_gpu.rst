Python interface (GPU)
======================

Quick-start examples
--------------------

As mentioned in the :ref:`Python GPU installation instructions <install-python-gpu>`, the easiest way to install the Python interface for cuFINUFFT is to run ``pip install cufinufft``.

*Note*: The interface to cuFINUFFT has changed between versions 1.3 and 2.2.
Please see :ref:`Migration to cuFINUFFT v2.2<cufinufft_migration>` for details.

Assuming cuFINUFFT has been installed, we will now consider how to calculate a 1D type 1 transform.
To manage the GPU and transfer to and from host and device, we will use the ``cupy`` library.
Consequently, we start with a few import statements.

.. code-block:: python

    import cupy as cp

    import cufinufft

We then proceed to setting up a few parameters.

.. code-block:: python

    # number of nonuniform points
    M = 100000

    # grid size
    N = 200000

    # generate positions for the nonuniform points and the coefficients
    x_gpu = 2 * cp.pi * cp.random.uniform(size=M)
    c_gpu = (cp.random.standard_normal(size=M)
             + 1J * cp.random.standard_normal(size=M))

Now that the data is prepared, we can call cuFINUFFT to compute the transform from the ``M`` source points onto a grid of ``N`` modes

.. code-block:: python

    f_gpu = cufinufft.nufft1d1(x_gpu, c_gpu, (N,))

    # move results off the GPU
    f = f_gpu.get()

The last line is needed here to copy the data from the device (GPU) to the host (CPU).
Now ``f`` is a size-``N`` array containing the NUDFT of ``c`` at the points ``x``.
This example, along with similar examples for other frameworks, can be found in ``python/cufinufft/examples/getting_started_*.py``.

Other possible calculations are possible by supplying different options during plan creation.
See the full API documentation below for more information.

Full documentation
------------------

.. automodule:: cufinufft
    :members:
    :member-order: bysource
    :undoc-members:
