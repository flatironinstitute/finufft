MATLAB/octave interfaces
========================

Quick-start example
~~~~~~~~~~~~~~~~~~~

To demo a single 1D transform of type 1 (nonuniform points to uniform
Fourier coefficients), we set up random data then do the transform
as follows:

.. code-block:: matlab

  M = 1e5;                            % number of NU source points
  x = 2*pi*rand(M,1);                 % points in a 2pi-periodic domain
  c = randn(M,1)+1i*randn(M,1);       % iid random complex data (row or col vec)
  N = 2e5;                            % how many desired Fourier modes?
  f = finufft1d1(x,c,+1,1e-12,N);     % do it (takes around 0.04 sec)

The column vector output ``f`` should be interpreted as the Fourier
coefficients with frequency indices ``k = -N/2:N/2-1``.
(This is because ``N`` is even; otherwise ``k = -(N-1)/2:(N-1)/2``.)
The values in ``f`` are accurate (relative to this vector's 2-norm)
to roughly 12 digits, as requested by the tolerance argument ``1e-12``.
Choosing a larger (ie, worse) tolerance leads to faster transforms.
The ``+1`` controls the sign in the exponential in the formula, that is
being evaluated; recall this is (1) on the `front page<index>`.

The above we call the "simple interface". There is also a "vectorized"
interface which does the same but with multiple stacked strength vectors.
We demo this reusing ``x`` and ``N`` from above:

.. code-block:: matlab

  Ntr = 1e2;                          % number of vectors (transforms to do)
  C = randn(M,Ntr)+1i*randn(M,Ntr);   % iid random complex data (matrix)
  F = finufft1d1(x,C,+1,1e-12,N);     % do it (takes around 1 sec)





  *** Guru demo

  




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
The low-level :ref:`error number codes <errcodes>` are not used.

If you have added the ``matlab`` directory of FINUFFT correctly to your
MATLAB path via something like ``addpath FINUFFT/matlab``, then
``help finufft/matlab`` will give the summary of all commands:

.. literalinclude:: ../matlab/Contents.m

The individual commands have the following help documentation:
                    
.. include:: matlabhelp.raw
