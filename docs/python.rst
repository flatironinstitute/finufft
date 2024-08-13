Python interface
================

Quick-start examples
--------------------

The easiest way to install is to run::

  pip install finufft

which downloads and installs the latest precompiled binaries from PyPI.
If you would like to compile from source, you can tell ``pip`` to compile the library from source with the option ``--no-binary`` using the command::

  pip install --no-binary finufft finufft

By default, this will use the ``-march=native`` flag when compiling the library, which should result in improved performance.
Note that ``finufft`` has to be specified twice (first as an argument to ``--no-binary`` and second as the package the is to be installed). This option also allows you to switch out the default FFT library (FFTW) for DUCC0 using::

  pip install --no-binary finufft finufft --config-settings=cmake.define.FINUFFT_USE_DUCC0=ON finufft

If you have ``pytest`` installed, you can test it with::

  pytest python/finufft/test

or, without having ``pytest`` you can run the older-style eyeball check::

  python3 python/finufft/test/run_accuracy_tests.py

which should report errors around ``1e-6`` and throughputs around 1-10 million points/sec.
(Please note that the ``finufftpy`` package is obsolete.)
If you would like to compile from source, see :ref:`the Python installation instructions <install-python>`.

Once installed, to calculate a 1D type 1 transform from nonuniform to uniform points, we import ``finufft``, specify the nonuniform points ``x``, their strengths ``c``, and call ``nufft1d1``:

.. code-block:: python

    import numpy as np
    import finufft

    # number of nonuniform points
    M = 100000

    # the nonuniform points
    x = 2 * np.pi * np.random.uniform(size=M)

    # their complex strengths
    c = (np.random.standard_normal(size=M)
        + 1J * np.random.standard_normal(size=M))

    # desired number of Fourier modes (uniform outputs)
    N = 200000

    # calculate the transform
    f = finufft.nufft1d1(x, c, N)

The input here is a set of complex strengths ``c``, which are used to approximate (1) in :ref:`math`.
That approximation is stored in ``f``, which is indexed from ``-N // 2`` up to ``N // 2 - 1`` (since ``N`` is even; if odd it would be ``-(N - 1) // 2`` up to ``(N - 1) // 2``).
The approximation is accurate to a tolerance of ``1e-6``, which is the default tolerance of ``nufft1d1``.
It can be modified using the ``eps`` argument:

.. code-block:: python

    # calculate the transform to higher accuracy
    f = finufft.nufft1d1(x, c, N, eps=1e-12)

Note, however, that a lower tolerance (that is, a higher accuracy) results in a slower transform. See ``python/finufft/examples/simple1d1.py`` for the demo code that includes a basic math test (useful to check both the math and the indexing).

For higher dimensions, we would specify point locations in more than one dimension:

.. code-block:: python

    # 2D nonuniform points (x,y coords)
    x = 2 * np.pi * np.random.uniform(size=M)
    y = 2 * np.pi * np.random.uniform(size=M)

    # desired number of Fourier modes (in x, y directions respectively)
    N1 = 1000
    N2 = 2000

    # the 2D transform outputs f array of shape (N1, N2)
    f = finufft.nufft2d1(x, y, c, (N1, N2))

See ``python/finufft/examples/simple2d1.py`` for the demo code that includes a basic math test (useful to check both the math and the indexing).

We can also go the other way, from uniform to non-uniform points, using a type 2 transform:

.. code-block:: python

    # input Fourier coefficients
    f = (np.random.standard_normal(size=(N1, N2))
         + 1J * np.random.standard_normal(size=(N1, N2)))

    # calculate the 2D type 2 transform
    c = finufft.nufft2d2(x, y, f)

Now the output is a complex vector of length ``M`` approximating (2) in :ref:`math`, that is the adjoint (but not inverse) of (1). (Note that the default sign in the exponential is negative for type 2 in the Python interface.)

In addition to tolerance ``eps``, we can adjust other options for the transform.
These are listed in :ref:`opts` and are specified as keyword arguments in the Python interface.
For example, to change the mode ordering to FFT style (that is, in each dimension ``Ni = N1`` or ``N2``, the indices go from ``0`` to ``Ni // 2 - 1``, then from ``-Ni // 2`` to ``-1``, since each ``Ni`` is even), we call

.. code-block:: python

    f = finufft.nufft2d1(x, y, c, (N1, N2), modeord=1)

We can also specify a preallocated output array using the ``out`` keyword argument.
This would be done by

.. code-block:: python

    # allocate the output array
    f = np.empty((N1, N2), dtype='complex128')

    # calculate the transform
    finufft.nufft2d1(x, y, c, out=f)

In this case, we do not need to specify the output shape since it can be inferred from ``f``.

Note that the above functions are all vectorized, which means that they can take multiple inputs stacked along the first dimension (that is, in row-major order) and process them simultaneously.
This can bring significant speedups for small inputs by avoiding multiple short calls to FINUFFT.
For the 2D type 1 vectorized interface, we would call

.. code-block:: python

    # number of transforms
    K = 4

    # generate K stacked coefficient arrays
    c = (np.random.standard_normal(size=(K, M))
         + 1J * np.random.standard_normal(size=(K, M)))

    # calculate the K transforms simultaneously (K is inferred from c.shape)
    f = finufft.nufft2d1(x, y, c, (N1, N2))

The output array ``f`` would then have the shape ``(K, N1, N2)``.
See the complete demo in ``python/finufft/examples/many2d1.py``.

More fine-grained control can be obtained using the plan (or `guru`) interface.
Instead of preparing the transform, setting the nonuniform points, and executing the transform all at once, these steps are seperated into different function calls.
This can speed up calculations if multiple transforms are executed for the same grid size, since the same FFTW plan can be reused between calls.
Additionally, if the same nonuniform points are reused between calls, we gain an extra speedup since the points only have to be sorted once.
To perform the call above using the plan interface, we would write

.. code-block:: python

    # specify type 1 transform
    nufft_type = 1

    # instantiate the plan (note ntrans must be set here)
    plan = finufft.Plan(nufft_type, (N1, N2), n_trans=K)

    # set the nonuniform points
    plan.setpts(x, y)

    # execute the plan
    f = plan.execute(c)

See the complete demo in ``python/finufft/examples/guru2d1.py``.
All interfaces support both single and double precision, but for the plan, this must be specified at initialization time using the ``dtype`` argument

.. code-block:: python

    # convert input data to single precision
    x = x.astype('float32')
    y = y.astype('float32')
    c = c.astype('complex64')

    # instantiate the plan and set the points
    plan = finufft.Plan(nufft_type, (N1, N2), n_trans=K, dtype='complex64')
    plan.setpts(x, y)

    # execute the plan, giving single-precision output
    f = plan.execute(c)

See the complete demo, with math test, in ``python/finufft/examples/guru2d1f.py``.


Full documentation
------------------

.. automodule:: finufft
    :members:
    :member-order: bysource
