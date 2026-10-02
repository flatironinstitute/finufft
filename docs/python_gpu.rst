Python interface (GPU)
======================

Quick-start examples
--------------------

As mentioned in the :ref:`Python GPU installation instructions <install-python-gpu>`, the easiest way to install the Python interface for cuFINUFFT is to run::

  pip install cufinufft

If you would like to compile from source, you can tell ``pip`` to compile the library from source with the option ``--no-binary`` using the command::

  pip install --no-binary cufinufft cufinufft

This will often result in improved performance since the build will be optimized for your particular architecture.
In particular, it can take advantage of newer CUDA features if you have a recent version of CUDA installed.
Note that ``cufinufft`` has to be specified twice (first as an argument to ``--no-binary`` and second as the package that is to be installed).

*Note*: The interface to cuFINUFFT has changed between versions 1.3 and 2.2.
Please see :ref:`Migration to cuFINUFFT v2.2<cufinufft_migration>` for details.

Assuming cuFINUFFT has been installed, we will now consider how to calculate a 1D type 1 transform.
To manage the GPU and transfer to and from host and device, we will use the ``cupy`` library.
Here is the complete demo (``python/cufinufft/examples/getting_started_cupy.py``):

.. literalinclude:: ../python/cufinufft/examples/getting_started_cupy.py
  :language: python
  :start-after: docs-start: getting-started-cupy
  :end-before: docs-end: getting-started-cupy

The last line is needed here to copy the data from the device (GPU) to the host (CPU).
Now ``f`` is a size-``N`` array containing the NUDFT of ``c`` at the points ``x``.
This example, along with similar examples for other frameworks, can be found in ``python/cufinufft/examples/getting_started_*.py``.

Other possible calculations are possible by supplying different options during plan creation.
See the full API documentation below for more information.

CUDA streams
------------

``cufinufft.Plan`` accepts ``gpu_stream=<stream pointer>`` (for example
``cupy.cuda.Stream().ptr``) to run the plan on a given stream; the semantics and
caveats are those of the C interface, see :ref:`CUDA streams<gpu_streams>`.
Since the plan holds a raw, non-reference-counted handle, keep the stream object
alive and delete the ``Plan`` before it.
A double-buffered example is in
``python/cufinufft/examples/example3d2many_async_cupy.py``.

Full documentation
------------------

.. automodule:: cufinufft
    :members:
    :member-order: bysource
    :undoc-members:
    :exclude-members: Plan

.. autoclass:: cufinufft.Plan
    :members:
    :member-order: bysource
    :undoc-members:
