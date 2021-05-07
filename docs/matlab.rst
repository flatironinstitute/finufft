MATLAB/octave interfaces
========================

.. note::

   See the :ref:`installation page <install>` for how to build these interfaces, or look `here <http://users.flatironinstitute.org/~ahb/codes/finufft-binaries>`_.

Quick-start examples
~~~~~~~~~~~~~~~~~~~~

To demo a single 1D transform of type 1 (nonuniform points to uniform
Fourier coefficients), we set up random data then do the transform
as follows:

.. code-block:: matlab

  M = 1e5;                            % number of NU source points
  x = 2*pi*rand(M,1);                 % points in a 2pi-periodic domain
  c = randn(M,1)+1i*randn(M,1);       % iid random complex data (row or col vec)
  N = 2e5;                            % how many desired Fourier modes?
  f = finufft1d1(x,c,+1,1e-12,N);     % do it (takes around 0.02 sec)

The column vector output ``f`` should be interpreted as the Fourier
coefficients with frequency indices ``k = -N/2:N/2-1``.
(This is because ``N`` is even; otherwise ``k = -(N-1)/2:(N-1)/2``.)
The values in ``f`` are accurate (relative to this vector's 2-norm)
to roughly 12 digits, as requested by the tolerance argument ``1e-12``.
Choosing a larger (ie, worse) tolerance leads to faster transforms.
The ``+1`` controls the sign in the exponential; recall equation
:eq:`1d1`. All :ref:`options<opts>` maybe be changed from
their defaults, for instance:

.. code-block:: matlab

  o.modeord = 1;                      % choose FFT-style output mode ordering  
  f = finufft1d1(x,c,+1,1e-12,N,o);   % do it

The above usage we call the "simple" interface. There is also a "vectorized"
interface which does the transform for multiple stacked strength vectors,
using the same nonuniform points each time.
We demo this, reusing ``x`` and ``N`` from above:

.. code-block:: matlab

  ntr = 1e2;                          % number of vectors (transforms to do)
  C = randn(M,ntr)+1i*randn(M,ntr);   % iid random complex data (matrix)
  F = finufft1d1(x,C,+1,1e-12,N);     % do them (takes around 1.2 sec)

Here this is nearly twice as fast as doing 100 separate calls to the simple
interface. For smaller transform sizes the acceleration factor of this vectorized call can be much higher.

If you want yet more control, consider using the "guru" interface.
This can be faster than fresh calls to the simple or vectorized interfaces
for the same number of transforms, for reasons such as this:
the nonuniform points can be changed between transforms, without forcing
FFTW to look up a previously stored plan.
Usually, such an acceleration is only important when doing
repeated small transforms, where "small" means each transform takes of
order 0.01 sec or less.
Here we use the guru interface to repeat the first demo above:

.. code-block:: matlab

  type = 1; ntr = 1; o.modeord = 1;   % transform type, #transforms, opts
  N = 2e5;                            % how many desired Fourier modes?
  plan = finufft_plan(1,N,+1,ntr,1e-12,o);      % plan for N output modes
  M = 1e5;                            % number of NU source points
  x = 2*pi*rand(M,1);                 % array of NU source points
  plan.setpts(x,[],[]);               % pass pointer to this array (M inferred)
  % (note: the x array should now not be altered until all executes are done!)
  c = randn(M,1)+1i*randn(M,1);       % iid random complex data (row or col vec)
  f = plan.execute(c);                % do the transform (0.008 sec, ie, faster)
  % ...one could now change the points with setpts, and/or do new transforms
  % with new c data...
  delete(plan);                       % don't forget to clean up

.. warning::
     
   If an existing array is passed to ``setpts``, then this array must not be altered before ``execute`` is called! This is because, in order to save RAM (allowing larger problems to be solved), internally FINUFFT stores only *pointers* to ``x`` (etc), rather than unnecessarily duplicating this data. This is not true if an *expression* such as ``-x`` or ``2*pi*rand(M,1)`` is passed to ``setpts``, since in those cases the ``plan`` object does make internal copies, as per MATLAB's usual shallow-copy argument passing.

Finally, we demo a 2D type 1 transform using the simple interface. Let's
request a rectangular Fourier mode array of 1000 modes in the x direction but 500 in the
y direction. The source points are in the square of side length $2\pi$:

.. code-block:: matlab

  M = 1e6; x = 2*pi*rand(1,M); y = 2*pi*rand(1,M);     % points in [0,2pi]^2
  c = randn(M,1)+1i*randn(M,1);       % iid random complex data (row or col vec)
  N1 = 1000; N2 = 500;                % desired Fourier mode array sizes
  f = finufft2d1(x,y,c,+1,1e-9,N1,N2);          % do it (takes around 0.08 sec)

The resulting output ``f`` is indeed size 1000 by 500. The first dimension
(number of rows) corresponds to the x input coordinate, and the second to y.

If you need to change the definition of the period from $2\pi$, simply
linearly rescale your points before sending them to FINUFFT.

.. note::

   Under the hood FINUFFT has double- and single-precision libraries.
   The simple and vectorized MATLAB/octave interfaces infer which to call by checking the class of its input arrays, which must all match (ie, all must be ``double`` or all must be ``single``).
   Since by default MATLAB arrays are double-precision, this is the precision that all of the above examples run in.
   To perform single-precision transforms, send in single-precision data.
   In contrast, precision in the guru interface is set with the ``finufft_plan`` option string ``o.floatprec``, either ``'double'`` (the default), or ``'single'``.

See
`tests and examples in the repo <https://github.com/flatironinstitute/finufft/tree/master/matlab/>`_ and
:ref:`tutorials and demos<tut>` for plenty more MATLAB examples.

Full documentation
~~~~~~~~~~~~~~~~~~

Here are the help documentation strings for all MATLAB/octave interfaces.
They only abbreviate the options (for full documentation see :ref:`opts`).
Informative warnings and errors are raised in MATLAB style with unique
codes (see ``../matlab/errhandler.m``, ``../matlab/finufft.mw``, and
``../valid_*.m``).
The low-level :ref:`error number codes <error>` are not used.

If you have added the ``matlab`` directory of FINUFFT correctly to your
MATLAB path via something like ``addpath FINUFFT/matlab``, then
``help finufft/matlab`` will give the summary of all commands:

.. literalinclude:: ../matlab/Contents.m

The individual commands have the following help documentation:
                    
.. include:: matlabhelp.doc
