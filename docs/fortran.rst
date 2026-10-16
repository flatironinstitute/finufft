.. _fort:

Usage from Fortran
==========================

We provide Fortran interfaces that are very similar to those in C/C++.
We deliberately use "legacy" Fortran style (in the `terminology
of FFTW <http://www.fftw.org/fftw3_doc/Calling-FFTW-from-Legacy-Fortran.html>`_), enabling the widest applicability and avoiding the complexity of
later Fortran features.
Namely, we use f77, with two features from f90: dynamic allocation
and derived types. The latter is only needed if options must be
changed from default values.
We also include, listed at the bottom below, a "modern" f90 demo using a module.

Quick-start example
~~~~~~~~~~~~~~~~~~~~~~

To perform a double-precision 1D type 1 transform from ``M`` nonuniform points ``xj``
with strengths ``cj``, to ``N`` output modes whose coefficients will be written
into the ``fk`` array, using 9-digit tolerance, the $+i$ imaginary sign,
and default options, the declarations and call are

.. code-block:: fortran

      include 'finufft.fh'

      integer ier,iflag
      integer*8 N,M
      real*8, allocatable :: xj(:)
      real*8 tol
      complex*16, allocatable :: cj(:),fk(:)
      type(finufft_opts) opts
      type(finufft_opts), pointer :: defopts => null()

 !    (...allocate xj, cj, and fk, and fill xj and cj here...)

      tol = 1.0D-9
      iflag = +1
      call finufft1d1(M,xj,cj,iflag,tol,N,fk,defopts,ier)

which writes the output to ``fk``, and the status to the integer ``ier``.
Since the default is CMCL mode ordering, the output for frequency index ``k``
is found in ``fk(k+N/2+1)``.
``ier=0`` indicates success, otherwise error codes are
as in :ref:`here <error>`.
By default (``opts.nthreads=0``), the number of physical cores available is used (honoring ``OMP_NUM_THREADS`` if set), unless FINUFFT was built single-threaded; see :ref:`opts`.
(Note that here the disassociated pointer ``defopts`` is simply a way to pass
a NULL pointer to our C++ wrapper; another would be ``%val(0_8)``.)
For a minimally complete test code demonstrating the above see
`simple1d1.f <https://github.com/flatironinstitute/finufft/blob/master/fortran/examples/simple1d1.f>`_.

.. note::

   Higher-dimensional arrays are stored in Fortran ordering
   with $x$ (``N1``) the fastest direction, and, in the vectorized
   ("many") calls, the transform number is slowest (transforms are
   stacked not interleaved).
   For instance, for the 2D type 1 vectorized transform
   ``finufft2d1many(ntrans,M,xj,yj,cj,iflag,tol,N1,N2,fk,opts,ier)``
   with CMCL mode-ordering,
   the ``(k1,k2)`` frequency coefficient from transform number ``t`` is
   to be found at ``fk(k1+N1/2+1 + (k2+N2/2)*N1 + t*N1*N2)``.

Building and running the examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the top-level directory of the repository, configure with the Fortran
wrappers and examples enabled, build, and run an example
(each ``fort_*`` program is named after its source file)::

  cmake -S . -B build -DFINUFFT_BUILD_FORTRAN=ON -DFINUFFT_BUILD_EXAMPLES=ON
  cmake --build build
  build/fortran/fort_simple1d1

(The ``fortran`` CMake preset does the same, but with the DUCC0 FFT library
and the build directory ``build/fortran``.)
On success each example prints one line with its relative error.
On a wrong result it prints one ``FAILED`` message and exits with a nonzero status.

To compile and link a program of your own against the FINUFFT static library
from this build, one must list dependent libraries by hand.
From the ``fortran/examples/`` directory, using GCC on Linux::

  gfortran -fopenmp -I../../include simple1d1.f -o simple1d1 \
    ../../build/src/libfinufft.a \
    ../../build/src/common/libfinufft_common.a \
    -lfftw3 -lfftw3_omp -lfftw3f -lfftw3f_omp -lstdc++ -lm

Then to execute run ``./simple1d1``.
The demos ``nufft*d_demo*.f`` also need the direct summation routine, for
instance by adding ``../directft/dirft1d.f`` (or ``dirft1df.f`` for single precision) to the compile line.
In Mac OSX, replace ``fftw3_omp`` by ``fftw3_threads``, and if you use
clang, replace ``-fopenmp`` by ``-Xclang -fopenmp`` and link ``-lomp``.

.. note ::
 Our simple interface is designed to be a near drop-in replacement for the native f90 `CMCL libraries of Greengard-Lee <http://www.cims.nyu.edu/cmcl/nufft/nufft.html>`_. The differences are: i) we added a penultimate argument in the list which allows options to be changed, and ii) our normalization differs for type 1 transforms (divide FINUFFT output by $M$ to match CMCL output).

Changing options
~~~~~~~~~~~~~~~~

To choose non-default options in the above example, create an options
derived type, set it to default values, change whichever you wish, and pass
it to FINUFFT. This is what the second half of the
``fortran/examples/simple1d1.f`` demo does:

.. literalinclude:: ../fortran/examples/simple1d1.f
  :language: fortran
  :start-after: docs-start: options
  :end-before: docs-end: options

The full example is ``fortran/examples/simple1d1.f``. From the ``fortran/examples/`` directory, compile it with ``gfortran -fopenmp -I../../include simple1d1.f -o simple1d1 ../../build/src/libfinufft.a ../../build/src/common/libfinufft_common.a -lfftw3 -lfftw3_omp -lfftw3f -lfftw3f_omp -lstdc++ -lm`` and run it with ``./simple1d1``.

The same demo in "modern" f90 style, using the ``finufft_mod`` module instead
of the include file (see ``fortran/examples/simple1d1.f90``), sets up the
transform like this (again minus the accuracy check):

.. literalinclude:: ../fortran/examples/simple1d1.f90
  :language: fortran
  :start-after: docs-start: simple1d1-f90-setup
  :end-before: docs-end: simple1d1-f90-setup

The full example is ``fortran/examples/simple1d1.f90``. From the ``fortran/examples/`` directory, compile it with ``gfortran -fopenmp -I../../include ../../include/finufft_mod.f90 simple1d1.f90 -o simple1d1f90 ../../build/src/libfinufft.a ../../build/src/common/libfinufft_common.a -lfftw3 -lfftw3_omp -lfftw3f -lfftw3f_omp -lstdc++ -lm`` and run it with ``./simple1d1f90``.

and its options-changing second half is:

.. literalinclude:: ../fortran/examples/simple1d1.f90
  :language: fortran
  :start-after: docs-start: simple1d1-f90-options
  :end-before: docs-end: simple1d1-f90-options

The full example is ``fortran/examples/simple1d1.f90``. Compile it with the ``gfortran`` command that the previous sentence gives, and run it with ``./simple1d1f90``.

See ``modeord`` in :ref:`Options<opts>`
to instead use FFT-style mode ordering, which
simply differs by an ``fftshift`` (as it is commonly called).


Summary of Fortran interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The names of routines and the meanings of all arguments is identical
to the :ref:`C/C++ routines <c>`.
Eg, ``finufft2d3`` means double-precision 2D transform of type 3.
``finufft2d3many`` means applying double-precision
2D transforms of type 3 to a stack of many
strength vectors (vectorized interface).
``finufft2d3f`` means single-precision 2D type 3.
The guru interface has very similar arguments to its C/C++ version.
Compared to C/C++, all argument lists have ``ier`` appended at the end,
to which the status is written; this is the same as the return value
in the C/C++ interfaces.
These routines and arguments are, in double-precision:

.. code-block:: fortran

      include 'finufft.fh'
 !    (or in F90 one may instead "use finufft_mod")

      integer ier,iflag,ntrans,type,dim
      integer*8 M,N1,N2,N3,Nk
      integer*8 plan,n_modes(3)
      real*8, allocatable :: xj(:),yj(:),zj(:), sk(:),tk(:),uk(:)
      real*8 tol
      complex*16, allocatable :: cj(:), fk(:)
      type(finufft_opts) opts

 !    simple interface
      call finufft1d1(M,xj,cj,iflag,tol,N1,fk,opts,ier)
      call finufft1d2(M,xj,cj,iflag,tol,N1,fk,opts,ier)
      call finufft1d3(M,xj,cj,iflag,tol,Nk,sk,fk,opts,ier)
      call finufft2d1(M,xj,yj,cj,iflag,tol,N1,N2,fk,opts,ier)
      call finufft2d2(M,xj,yj,cj,iflag,tol,N1,N2,fk,opts,ier)
      call finufft2d3(M,xj,yj,cj,iflag,tol,Nk,sk,tk,fk,opts,ier)
      call finufft3d1(M,xj,yj,zj,cj,iflag,tol,N1,N2,N3,fk,opts,ier)
      call finufft3d2(M,xj,yj,zj,cj,iflag,tol,N1,N2,N3,fk,opts,ier)
      call finufft3d3(M,xj,yj,zj,cj,iflag,tol,Nk,sk,tk,uk,fk,opts,ier)

 !    vectorized interface
      call finufft1d1many(ntrans,M,xj,cj,iflag,tol,N1,fk,opts,ier)
      call finufft1d2many(ntrans,M,xj,cj,iflag,tol,N1,fk,opts,ier)
      call finufft1d3many(ntrans,M,xj,cj,iflag,tol,Nk,sk,fk,opts,ier)
      call finufft2d1many(ntrans,M,xj,yj,cj,iflag,tol,N1,N2,fk,opts,ier)
      call finufft2d2many(ntrans,M,xj,yj,cj,iflag,tol,N1,N2,fk,opts,ier)
      call finufft2d3many(ntrans,M,xj,yj,cj,iflag,tol,Nk,sk,tk,fk,opts,ier)
      call finufft3d1many(ntrans,M,xj,yj,zj,cj,iflag,tol,N1,N2,N3,fk,opts,ier)
      call finufft3d2many(ntrans,M,xj,yj,zj,cj,iflag,tol,N1,N2,N3,fk,opts,ier)
      call finufft3d3many(ntrans,M,xj,yj,zj,cj,iflag,tol,Nk,sk,tk,uk,fk,opts,ier)

 !    guru interface
      call finufft_makeplan(type,dim,n_modes,iflag,ntrans,tol,plan,opts,ier)
      call finufft_setpts(plan,M,xj,yj,zj,Nk,sk,yk,uk,ier)
      call finufft_execute(plan,cj,fk,ier)
      call finufft_execute_adjoint(plan,cj,fk,ier)
      call finufft_destroy(plan,ier)

The single-precision (ie, ``real*4`` and ``complex*8``)
functions are identical except with the replacement
of ``finufft`` with ``finufftf`` in each function name.
All are defined (from the C++ side) in ``fortran/finufftfort.cpp``.


Code examples
~~~~~~~~~~~~~

The ``fortran/examples`` directory contains the following demos,
mostly in both precisions.
Each has a math test to check the correctness of some or all outputs::

  simple1d1.f        - 1D type 1, simple interface, default and various opts
  guru1d1.f          - 1D type 1, guru interface, default and various opts
  guru1d1_adjoint.f  - adjoint of 1D type 1, guru interface, default opts
  guru1d2_adjoint.f  - adjoint of 1D type 2, guru interface, default and various opts
  nufft1d_demo.f     - 1D types 1,2,3, minimally changed from CMCL demo codes
  nufft2d_demo.f     - 2D "
  nufft3d_demo.f     - 3D "
  nufft2dmany_demo.f - 2D types 1,2,3, vectorized (many strengths) interface
  simple1d1.f90      - modern Fortran90 version of simple1d1 using module

These are the double-precision file names; the single precision have a
suffix ``f`` before the ``.f`` (apart from the f90 which has no single-precision
version).
The last four here are modified from demos in the
`CMCL NUFFT libraries <http://www.cims.nyu.edu/cmcl/nufft/nufft.html>`_.
The first three of these have been changed only to use FINUFFT.
The last four demos require direct summation (slow) reference implementations
of the transforms in ``fortran/directft``, modified from their CMCL
counterparts only to remove the $1/M$ prefactor for type 1 transforms.

To build and run all demos see above.

For authorship and licensing of the Fortran wrappers, see
the `README <https://github.com/flatironinstitute/finufft/blob/master/fortran/README>`_ in the fortran directory.
