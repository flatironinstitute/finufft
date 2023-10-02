.. _c:

Documentation of all C++ functions
==================================

All functions have double-precision (``finufft``) and single-precision
(``finufftf``) versions. Do not forget this ``f`` suffix in the latter case.
We group the simple and vectorized interfaces together, by each of the
nine transform types (dimensions 1,2,3, and types 1,2,3).
The guru interface functions are defined at the end.
You will also want to refer to the :ref:`options<opts>`
and :ref:`error codes<error>`
which apply to all 46 routines.

A reminder on Fourier mode ordering;
see :ref:`modeord<modeord>`.
For example, if ``N1=8`` in a 1D type 1 or type 2 transform:

* if ``opts.modeord=0``: frequency indices are ordered ``-4,-3,-2,-1,0,1,2,3`` (CMCL ordering)

* if ``opts.modeord=1``: frequency indices are ordered ``0,1,2,3,-4,-3,-2,-1`` (FFT ordering)

The orderings are related by a "fftshift".
This holds for each dimension.
Multidimensional arrays are passed by a pointer to
a contiguous Fortran-style array, with the
"fastest" dimension x, then y (if present), then z (if present), then
transform number (if ``ntr>1``).
We do not use C/C++-style multidimensional arrays; this gives us the most
flexibility from several languages without loss of speed or memory
due to unnecessary array copying.


In all of the simple, vectorized, and plan functions below you may either pass ``NULL`` as the last options
argument to use default options, or a pointer to a valid ``finufft_opts`` struct.
In this latter case you will first need to create an options struct
then set default values by passing a pointer (here ``opts``) to the following::

 void finufft_default_opts(finufft_opts* opts)
 void finufftf_default_opts(finufft_opts* opts)
   
  Set values in a NUFFT options struct to their default values.

Be sure to use the first version for double-precision and the second for single-precision. You may then change options with, for example, ``opts->debug=1;``
and then pass ``opts`` to the below routines.

Simple and vectorized interfaces
--------------------------------

The "simple" interfaces (the first two listed in each block) perform
a single transform, whereas the "vectorized" (the last two listed in each block,
with the word "many" in the function name) perform ``ntr`` transforms with the same set of nonuniform points but stacked complex strengths or coefficients vectors.

.. note::

  The motivations for the vectorized interface (and guru interface, see below) are as follows. 1) It is more efficient to bin-sort the nonuniform points only once if there are not to change between transforms. 2) For small problems, certain start-up costs cause repeated calls to the simple interface to be slower than necessary.  In particular, we note that FFTW takes around 0.1 ms per thread to look up stored wisdom, which for small problems (of order 10000 or less input and output data) can, sadly, dominate the runtime.


1D transforms
~~~~~~~~~~~~~

.. include:: c1d.doc

2D transforms
~~~~~~~~~~~~~

.. include:: c2d.doc

3D transforms
~~~~~~~~~~~~~

.. include:: c3d.doc


.. _guru:
             
Guru plan interface
-------------------                   

This provides more flexibility than the simple or vectorized interfaces.
Any transform requires (at least)
calling the following four functions in order. However, within this
sequence one may insert repeated ``execute`` calls, or another ``setpts``
followed by more ``execute`` calls, as long as the transform sizes (and number of transforms ``ntr``) are
consistent with those that have been set in the ``plan`` and in ``setpts``.
Keep in mind that ``setpts`` retains *pointers* to the user's list of nonuniform points, rather than copying these points; thus the user must not change their nonuniform point arrays until after any ``execute`` calls that use them.

.. note::

  The ``plan`` object (type ``finufft{f}_plan``) is an opaque pointer; the public interface specifies no more details that that. Under the hood in our library the plan happens to point to a C++ object of type ``finufft{f}_plan_s``, whose internal details the library user should not attempt to access, nor to rely on.

.. include:: cguru.doc
                    
