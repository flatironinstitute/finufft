MATLAB GPU interfaces
=====================

.. note::

   See the :ref:`MATLAB GPU installation page <install-matlab-gpu>` for how to build these interfaces.

We follow the :ref:`MATLAB/Octave CPU tutorial page <matlab>`, which
you should to read first, to learn about the simple interfaces and
array indexing.
The key fact is that our CUDA FINUFFT (aka CUFINUFFT) MATLAB interface
acts on ``gpuArray`` objects for the main I/O data arrays, and this requires
MATLAB's Parallel Computing Toolbox. This is a commercial product;
we do not currently have an Octave solution for GPU.


Quick-start examples
~~~~~~~~~~~~~~~~~~~~

Here we jump straight into a
2D type 1 transform using the simple interface in single precision. Let's
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
Here see a throughput of about 0.5 billion points/sec (on my A6000).

.. note::

   Timing GPU functions in MATLAB is misleading when using plain ``tic`` and ``toc``, because of asynchronous computation: the ``toc`` is executed before the ``gpuArray`` function has actually completed. For correct timings, use the following idea:

   .. code-block:: matlab
		   
     dev = gpuDevice();
     tic
     f = cufinufft2d1(x,y,c,+1,tol,N1,N2);
     wait(dev)
     toc

If you need to change the definition of the period from $2\pi$, you cannot;
instead linearly rescale your points before sending them to FINUFFT.

.. note::

   Under the hood cuFINUFFT has double- and single-precision libraries.
   The simple and vectorized GPU MATLAB interfaces infer which to call by checking the class of its input arrays, which must all match (ie, all must be ``double gpuArrays`` or all must be ``single gpuArrays``).
   Since by default MATLAB ``gpuArrays`` are double-precision, this is the precision that all of the above examples run in.
   To perform single-precision transforms, send in single-precision data.
   In contrast, precision in the guru interface is set with the ``cufinufft_plan`` option string ``opts.floatprec``, either ``'double'`` (the default), or ``'single'``.

For examples of using the cuFINUFFT guru (plan) interface, see
`examples in the repo <https://github.com/flatironinstitute/finufft/tree/master/matlab/examples/cuda>`_.


Full documentation of GPU interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here are the help documentation strings for all MATLAB GPU interfaces.
The interfaces are the same as the GPU ones prepended with "cu".
The options descriptions are rather abbreviated;
for full documentation see :ref:`opts_gpu`.
Informative warnings and errors are raised in MATLAB style with unique
codes (see ``../matlab/errhandler.m``, ``../matlab/cufinufft.mw``, and
``../valid_*.m``).
The low-level :ref:`error number codes <error>` are not used.

The individual GPU commands have the full help documentation:
                    
.. include:: matlabgpuhelp.doc
