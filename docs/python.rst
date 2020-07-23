Python interface
================

Quick-start examples
--------------------

These Python interfaces are written by Libin Lu, with help from David Stein, Alex Barnett, and Joakim Andén.
The easiest way to install is to run ``pip install finufftpy``, which downloads and installs the latest precompiled binaries from PyPI.
If you would like to compile from source, see :ref:`the Python installation instructions <install-python>`.

To calculate a 1D type 1 transform, from nonuniform to uniform points, we import ``finufftpy``, specify the frequencies ``x``, the coefficients ``c``, and call ``nufft1d1``:

.. code-block:: python

    import numpy as np
    import finufftpy

    # number of nonuniform points
    M = 100000

    # frequencies
    x = 2 * np.pi * np.random.uniform(size=M)

    # coefficients
    c = (np.random.standard_normal(size=M)
         + 1J * np.random.standard_normal(size=M))

    # number of Fourier modes
    N = 200000

    # calculate the transform
    f = finufftpy.nufft1d1(x, c, N)

The input here is a set of complex coefficients ``c``, which are used to approximate (1) in :ref:`math`.
That approximation is stored in ``f``, which is indexed from ``-N // 2`` to ``N // 2 - 1``.
The approximation is accurate to a tolerance of ``1e-6``, which is the default tolerance of ``nufft1d1``.
It can be modified using the ``eps`` argument

.. code-block:: python

    # calculate the transform to higher accuracy
    f = finufftpy.nufft1d1(x, c, N, eps=1e-12)

Note, however, that a lower tolerance (that is, a higher accuracy) results in a slower transform.

For higher dimensions, we would specify frequencies in more than one dimension:

.. code-block:: python

    # frequencies
    x = 2 * np.pi * np.random.uniform(size=M)
    y = 2 * np.pi * np.random.uniform(size=M)

    # number of Fourier modes
    N = 2000

    # calculate the 2D transform
    f = finufftpy.nufft2d1(x, y, c, N, N)


We can also go the other way, from uniform to non-uniform points, using a type 2 transform:

.. code-block:: python

    # coefficients
    f = (np.random.standard_normal(size=(N, N))
         + 1J * np.random.standard_normal(size=(N, N)))

    # calcualate the 2D type 2 transform
    c = finufftpy.nufft2d2(x, y, f)

Now the output is a complex vector of length ``M`` approximating (2) in :ref:`math`, that is the adjoint of (1).

In addition to tolerance ``eps``, we can adjust other options for the transform.
These are listed in :ref:`opts` and are specified as keyword arguments in the Python interface.
For example, to change the mode ordering to FFT style (that is, from ``0`` to ``N // 2 - 1``, then from ``- N // 2`` to ``-1``), we call

.. code-block:: python

    f = finufftpy.nufft2d1(x, y, c, N, N, modeord=1)

Note that the above functions are all vectorized, which means that they can take multiple inputs stacked along the last dimension (that is, in column-major order) and process them simultaneously.
This can bring significant speedups for small inputs by avoiding multiple short calls to FINUFFT.
For the 2D type 1 interface, we would call

.. code-block:: python

    # number of transforms
    K = 4

    # generate K separate coefficient arrays
    c = (np.random.standard_normal(size=(M, K))
         + 1J * np.random.standard_normal(size=(M, K)))

    # calculate the K transforms simultaneously
    f = finufftpy.nufft2d1(x, y, c, N, N)

The output array ``f`` would then have the shape ``(N, N, K)``.

More fine-grained control can be obtained using the plan (or `guru`) interface.
Instead of preparing the transform, setting the nonuniform points, and executing the transform all at once, these steps are seperated into different function calls.
This can speed up calculations if multiple transforms are executed for the same grid size, since the same FFTW plan can be reused between calls.
Additionally, if the same nonuniform points are reused between calls, we gain an extra speedup since the points only have to be sorted once.
To perform the call above using the plan interface, we would write

.. code-block:: python

    # specify type 1 transform
    nufft_type = 1
    n_trans = K

    # instantiate the plan
    plan = finufftpy.Plan(nufft_type, (N, N), n_trans=n_trans)

    # set the nonuniform points
    plan.setpts(x, y)

    # execute the plan
    f = plan.execute(c)

Full documentation
------------------

.. autofunction:: finufftpy.nufft1d1
.. autofunction:: finufftpy.nufft1d2
.. autofunction:: finufftpy.nufft1d3
.. autofunction:: finufftpy.nufft2d1
.. autofunction:: finufftpy.nufft2d2
.. autofunction:: finufftpy.nufft2d3
.. autofunction:: finufftpy.nufft3d1
.. autofunction:: finufftpy.nufft3d2
.. autofunction:: finufftpy.nufft3d3
