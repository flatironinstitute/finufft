MATLAB/octave interfaces
========================

A note on integer sizes:
In Matlab/MEX, mwrap uses ``int`` types, so that output arrays can only
be <2^31.
However, input arrays >=2^31 have been tested, and while they don't crash,
they result in wrong answers (all zeros). This has yet to be fixed.
An updated version of mwrap may be needed.

For a full list of error codes see :ref:`errcodes`.

.. literalinclude:: matlabhelp.raw
		    
