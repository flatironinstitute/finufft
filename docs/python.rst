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
Note that ``finufft`` has to be specified twice (first as an argument to ``--no-binary`` and second as the package that is to be installed). This option also allows you to switch out the default FFT library (FFTW) for DUCC0 using::

  pip install --no-binary finufft finufft --config-settings=cmake.define.FINUFFT_USE_DUCC0=ON finufft

If you have ``pytest`` installed, you can test it with::

  pytest python/finufft/test

or, without having ``pytest`` you can run the older-style eyeball check::

  python3 python/finufft/test/run_accuracy_tests.py

which should report errors around ``1e-6`` and throughputs around 1-10 million points/sec.
(Please note that the ``finufftpy`` package is obsolete.)
If you would like to compile from source, see :ref:`the Python installation instructions <install-python>`.

Once installed, to calculate a 1D type 1 transform from nonuniform to uniform points,
we import ``finufft``, specify the nonuniform points ``x``, their strengths ``c``,
and call ``nufft1d1`` — as in this complete demo
(``python/finufft/examples/simple1d1.py``, minus its final accuracy check):

.. literalinclude:: ../python/finufft/examples/simple1d1.py
  :language: python
  :start-after: docs-start: simple1d1
  :end-before: docs-end: simple1d1

The input here is a set of complex strengths ``c``, which are used to approximate (1) in :ref:`math`.
That approximation is stored in ``f``, which is indexed from ``-N // 2`` up to ``N // 2 - 1`` (since ``N`` is even; if odd it would be ``-(N - 1) // 2`` up to ``(N - 1) // 2``).
The tolerance requested via the ``eps`` argument is a trade-off:
a lower tolerance (that is, a higher accuracy) results in a slower transform.
See ``python/finufft/examples/simple1d1.py`` for the full demo including a basic math test (useful to check both the math and the indexing).

On CPU, if ``eps`` is so small that FINUFFT knows the requested accuracy is unattainable,
the Python interface raises ``RuntimeError`` (status ``ier=26``) during plan creation
or ``setpts``. If you want FINUFFT to clamp to the best-achievable accuracy and proceed
instead, pass ``allow_eps_too_small=1``.

For higher dimensions, we would specify point locations in more than one dimension,
as in this complete demo (``python/finufft/examples/simple2d1.py``, minus its
accuracy check):

.. literalinclude:: ../python/finufft/examples/simple2d1.py
  :language: python
  :start-after: docs-start: simple2d1
  :end-before: docs-end: simple2d1

We can also go the other way, from uniform to non-uniform points, using a type 2 transform:

.. code-block:: python

    # input Fourier coefficients
    f = (np.random.standard_normal(size=(N1, N2))
         + 1J * np.random.standard_normal(size=(N1, N2)))

    # calculate the 2D type 2 transform; output is a complex vector of length M
    c = finufft.nufft2d2(x, y, f)

The output ``c`` approximates (2) in :ref:`math`, that is the adjoint (but not inverse) of (1). (Note that the default sign in the exponential is negative for type 2 in the Python interface.)

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
Several options can be passed this way; ``python/finufft/examples/simpleopts1d1.py``
demonstrates ``debug``, ``modeord`` and ``upsampfac`` in a complete runnable demo:

.. literalinclude:: ../python/finufft/examples/simpleopts1d1.py
  :language: python
  :start-after: docs-start: simpleopts1d1
  :end-before: docs-end: simpleopts1d1

Note that the above functions are all vectorized, which means that they can take multiple inputs stacked along the first dimension (that is, in row-major order) and process them simultaneously.
This can bring significant speedups for small inputs by avoiding multiple short calls to FINUFFT.
Here is a complete demo of the 2D type 1 vectorized interface
(``python/finufft/examples/many2d1.py``, minus its accuracy check):

.. literalinclude:: ../python/finufft/examples/many2d1.py
  :language: python
  :start-after: docs-start: many2d1
  :end-before: docs-end: many2d1

More fine-grained control can be obtained using the plan (or `guru`) interface.
Instead of preparing the transform, setting the nonuniform points, and executing the transform all at once, these steps are seperated into different function calls.
This can speed up calculations if multiple transforms are executed for the same grid size, since the same FFTW plan can be reused between calls.
Additionally, if the same nonuniform points are reused between calls, we gain an extra speedup since the points only have to be sorted once.
To perform the call above using the plan interface
(complete demo ``python/finufft/examples/guru2d1.py``, minus its accuracy check):

.. literalinclude:: ../python/finufft/examples/guru2d1.py
  :language: python
  :start-after: docs-start: guru2d1
  :end-before: docs-end: guru2d1

The plan's ``n_modes`` property reports the mode counts in the same
``(N1, N2, ...)`` order passed to the constructor above (``ndarray.shape``
order), not the reversed order the underlying C library uses internally.

All interfaces support both single and double precision, but for the plan, this must be specified at initialization time using the ``dtype`` argument.
A complete single-precision demo
(``python/finufft/examples/guru2d1f.py``, minus its accuracy check):

.. literalinclude:: ../python/finufft/examples/guru2d1f.py
  :language: python
  :start-after: docs-start: guru2d1f
  :end-before: docs-end: guru2d1f

As above, requesting an unattainable ``eps`` now raises ``RuntimeError`` by default.
For exploratory or backwards-compatible workflows that prefer clamp-and-proceed behavior,
pass ``allow_eps_too_small=1`` when constructing the plan or calling the simple interface.

See the complete demo, with math test, in ``python/finufft/examples/guru2d1f.py``.


The ``finufft`` package ships inline type annotations and a ``py.typed``
marker file, so type checkers such as ``mypy`` pick up its signatures
without a separate stub package.

Full documentation
------------------

.. automodule:: finufft
    :members:
    :member-order: bysource
