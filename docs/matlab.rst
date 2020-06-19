MATLAB/octave interfaces
========================

Quick-start example
~~~~~~~~~~~~~~~~~~~

To perform a single

The simple (single-vector) and many vector interfaces are combined,
meaning that 




See
`tests and examples in the repo <https://github.com/flatironinstitute/finufft/tree/master/matlab/>`_ and
:ref:`tutorials and demos<tut>` for plenty more MATLAB examples.

Full documentation
~~~~~~~~~~~~~~~~~~

Here are the help documentation strings for all MATLAB/octave interfaces.
They only abbreviate the options (for full documentation see :ref:`opts`).
Informative warnings and errors are raised in MATLAB style
(see ``../matlab/errhandler.m``).
The number codes in :ref:`errcodes` are not returned.

If you have added the ``matlab`` directory of FINUFFT correctly to your
MATLAB path via something like ``addpath FINUFFT/matlab``, then
``help finufft/matlab`` will give the summary of all commands:

.. literalinclude:: ../matlab/Contents.m

The individual commands have the following help documentation:
                    
.. include:: matlabhelp.raw
