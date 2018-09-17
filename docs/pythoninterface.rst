Python interface
================

These python interfaces are by Daniel Foreman-Mackey, Jeremy Magland, and Alex Barnett, with help from David Stein.
See the installation notes for how to install these interfaces; the main thing to remember is to compile the library before trying to `pip install`. Below is the documentation for the nine routines. The 2d1 and 2d2
"many vector" interfaces are now also included.

Notes:

  #. The module has been designed not to recompile the C++ library; rather, it links to the existing static library. Therefore this library must have been compiled before building python interfaces.

  #. In the below, "float" and "complex" refer to double-precision for the default library. One can compile the library for single-precision, but the python interfaces are untested in this case.

  #. NumPy input and output arrays are generally passed directly without copying, which helps efficiency in large low-accuracy problems. In 2D and 3D, copying is avoided when arrays are Fortran-ordered; hence choose this ordering in your python code if you are able (see ``python_tests/accuracy_speed_tests.py``).

  #. Fortran-style writing of the output to a preallocated NumPy input array is used. That is, such an array is treated as a pointer into which the output is written. This avoids creation of new arrays. The python call return value is merely a status indicator.

.. autofunction:: finufftpy.nufft1d1
.. autofunction:: finufftpy.nufft1d2
.. autofunction:: finufftpy.nufft1d3
.. autofunction:: finufftpy.nufft2d1
.. autofunction:: finufftpy.nufft2d1many
.. autofunction:: finufftpy.nufft2d2
.. autofunction:: finufftpy.nufft2d2many
.. autofunction:: finufftpy.nufft2d3
.. autofunction:: finufftpy.nufft3d1
.. autofunction:: finufftpy.nufft3d2
.. autofunction:: finufftpy.nufft3d3
