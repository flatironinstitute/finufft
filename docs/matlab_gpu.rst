MATLAB GPU interfaces
=====================

.. note::

   See the :ref:`MATLAB GPU installation page <install-matlab-gpu>` for how to build these interfaces.

We in 2025 we added MATLAB Parallel Computing Toolbox ``gpuArray`` interfaces to
CUFINUFFT, our GPU library.
Here we follow the :ref:`MATLAB/Octave CPU tutorial page <matlab>`, which
you should to read first, to learn the basics of the interfaces and
array indexing.
The key change for the GPU user
is that our CUDA FINUFFT (aka CUFINUFFT) MATLAB interface
acts on ``gpuArray`` objects for the main I/O data arrays, and this requires
MATLAB's Parallel Computing Toolbox. This is a commercial product,
and we do not currently have an Octave solution for the GPU.


Quick-start example
~~~~~~~~~~~~~~~~~~~

We jump straight into a
2D type 1 transform using the simple interface, in single precision. Let's
request a rectangular output Fourier mode array of 10000 modes in the x direction but 5000 in the
y direction. We create 100 millions source points directly on the GPU, with coordinates lying in the square of side length $2\pi$:

.. code-block:: matlab

  M = 1e8;
  x = 2*pi*gpuArray.rand(M,1,'single');   % random pts in [0,2pi]^2
  y = 2*pi*gpuArray.rand(M,1,'single');
  c = gpuArray.randn(M,1,'single')+1i*gpuArray.randn(M,1,'single');    % iid random complex data
  N1 = 10000; N2 = 5000;                   % desired Fourier mode array sizes
  tol = 1e-3;
  f = cufinufft2d1(x,y,c,+1,tol,N1,N2);    % do it (takes around 0.2 sec)

The resulting output ``f`` is a complex single-precision ``gpuArray`` of size
10000 by 5000. The first dimension
(number of rows) corresponds to the x input coordinate, and the second to y.
If you need to change the definition of the period from $2\pi$, you cannot;
instead linearly rescale your points before sending them to FINUFFT.
The above shows a throughput of about 0.5 billion points/sec (on my A6000).
For the full example code that also verifies one of the outputs,
see `simple1d1f_gpu.m <https://github.com/flatironinstitute/finufft/tree/master/matlab/examples/cuda/simple1d1f_gpu.m>`_.

.. note::

   Timing GPU functions in MATLAB is misleading when using plain ``tic`` and ``toc``, because of asynchronous computation: the ``toc`` is often executed before the ``gpuArray`` function has actually completed! For correct timings, use the following pattern:

   .. code-block:: matlab

     dev = gpuDevice();
     tic
     f = cufinufft2d1(x,y,c,+1,tol,N1,N2);
     wait(dev)
     toc

.. note::

   Under the hood cuFINUFFT has double- and single-precision libraries. The simple and vectorized GPU MATLAB interfaces infer which precision library to call by checking the precision of its input arrays, which must all match (ie, all must be ``double gpuArrays`` or all must be ``single gpuArrays``). The precision of a ``gpuArray`` cannot be inferred via ``class``; instead one can use ``underlyingType`` (introduced recently in R2020b). In contrast, precision in the guru interface is set with the ``cufinufft_plan`` option string ``opts.floatprec``, either ``'double'`` (the default), or ``'single'``.

.. warning::

   The ``cufinufft?d?`` MATLAB commands are a guaranteed way to call the GPU library. We also offer an **experimental GPU overloading** of the ``finufft?d?`` MATLAB commands: when the first argument ``x`` is a ``gpuArray`` then this redirects to the corresponding ``cufinufft?d?`` function. This follows The MathWorks style, and is achieved by the tiny wrapper codes in ``matlab/@gpuArray/``. (We actually exploit this in ``test/fullmathtest.m``.)  For this to work the user must add the FINUFFT ``matlab`` directory to the path *before any* ``gpuArray`` objects are used in the current session. This is because the ``gpuArray`` class is "closed". Since TMW may prevent library developers from overloading in this way at any time in the future, the overloaded interface remains experimental as of FINUFFT version 2.4.


For several examples of using the cuFINUFFT guru (plan) interface, see
`examples in the repo <https://github.com/flatironinstitute/finufft/tree/master/matlab/examples/cuda>`_.


Full documentation of GPU interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here are the help documentation strings for all MATLAB GPU interfaces.
The interfaces are the same as the GPU ones except preceded by "cu".
The options descriptions are rather abbreviated in the below;
for full documentation see :ref:`opts_gpu`.
Informative warnings and errors are raised in MATLAB style with unique
codes (see sources ``errhandler.m``, ``cufinufft.mw``, and
``valid_*.m`` found `here <https://github.com/flatironinstitute/finufft/tree/master/matlab/>`_).
The low-level :ref:`error number codes <error>` are not used.

The individual GPU commands have the full help documentation:

.. include:: matlabgpuhelp.doc
