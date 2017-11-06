Python interface to FINUFFT
===========================

These python interfaces are by Daniel Foreman-Mackey, Jeremy Magland, and Alex Barnett, with help from David Stein.
See the installation notes for how to install these interfaces. Below is the documentation for the nine routines.

Notes:

#. The module has been designed not to recompile the C++ library; rather, it
links to the existing static library. ``pybind11`` is no longer needed.

#. NumPy input and output arrays are generally passed directly without copying, which helps efficiency in large low-accuracy problems. In 2D and 3D, copying is avoided when arrays are Fortran-ordered; hence choose this ordering in your python code if you are able (see ``python_tests/accuracy_speed_tests.py``).

#. Fortran-style writing of the output array to a preallocated NumPy array is used. That is, the output array is treated as a pointer into which the output is written. This avoids creation of new arrays. The python call return value is merely a status indicator.


.. todo:: Fix indentation docstrings, even though they look fine
      
.. autofunction:: finufftpy.finufft1d1
.. autofunction:: finufftpy.finufft1d2
.. autofunction:: finufftpy.finufft1d3
.. autofunction:: finufftpy.finufft2d1
.. autofunction:: finufftpy.finufft2d2
.. autofunction:: finufftpy.finufft2d3
.. autofunction:: finufftpy.finufft3d1
.. autofunction:: finufftpy.finufft3d2
.. autofunction:: finufftpy.finufft3d3
