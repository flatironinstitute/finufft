.. _fort:

Using FINUFFT from Fortran
==========================

We provide Fortran interfaces that are very similar to those in C/C++.
We deliberately use "legacy" Fortran style (in the terminology
of FFTW, see http://www.fftw.org/fftw3_doc/Calling-FFTW-from-Legacy-Fortran.html
), enabling the widest applicability and avoiding the complexity of
later fortran features.
Namely, we use f77, with two features from f90: dynamic allocation
and derived types. The former feature is only used for convenience in
the example drivers, and the latter is only needed if options must be
changed from default values.

Simple example
~~~~~~~~~~~~~~

To perform a 1D type 1 transform from ``M`` nonuniform points ``xj``
with strengths ``cj``, to ``N`` output modes whose coefficients will be written
into the ``fk`` array, using 9-digit tolerance, the $+i$ imaginary sign,
and default options, the declarations and call are

::

      integer ier,iflag
      integer*8 N,M
      real*8, allocatable :: xj(:)
      real*8 tol
      complex*16, allocatable :: cj(:),fk(:)
      integer*8, allocatable :: null

      (...allocate xj, cj, and fk, and fill xj and cj here...)

      tol = 1.0D-9
      iflag = +1
      call finufft1d1(M,xj,cj,iflag,tol,N,fk,null,ier)

which writes the output to ``fk``, and the status to the integer ``ier``.
A status 0 indicates success, otherwise error codes are
in :ref:`here <error>`.
All available OMP threads are used, unless FINUFFT was built single-thread.
(Note that here the unallocated ``null`` is simply a way to pass
a NULL pointer to our C++ wrapper; another would be ``%val(0_8)``.)
For a minimally complete test code demonstrating the above see
``fortran/examples/simple1d1.f``.

To compile (eg using GCC) and link such a program against FINUFFT::

  gfortran -I $(FINUFFT)/include simple1d1.f -o simple1d1 $(FINUFFT)/lib/libfinufft.so -lfftw3 -lfftw3_omp -lgomp -lstdc++

where ``$(FINUFFT)`` indicates the top-level FINUFFT directory.
Alternatively you may want to compile with ``g++`` and use ``-lgfortran`` at the *end* of the compile statement.
In Mac OSX, replace ``fftw3_omp`` by ``fftw3_threads``, and if you use
clang, ``-lgomp`` by ``-lomp``. See ``makefile`` and ``make.inc.*``.

.. note ::
 Our simple interface is designed to be a near drop-in replacement for the native f90 `CMCL libraries of Greengard-Lee <http://www.cims.nyu.edu/cmcl/nufft/nufft.html>`_. The differences are: i) we added a penultimate argument in the list which allows options to be changed, and ii) our normalization differs for type 1 transforms (one divides our output by $M$ to match the CMCL output).

Changing options
~~~~~~~~~~~~~~~~

To choose non-default options in the above example, create an options
derived type, set it to default values, change whichever you wish, and pass
it to FINUFFT, for instance

::
   
      include 'finufft.fh'
      type(nufft_opts) opts
 
      (...do stuff as before...)

      call finufft_default_opts(opts)
      opts%debug = 2
      opts%upsampfac = 1.25d0
      call finufft1d1(M,xj,cj,iflag,tol,N,fk,opts,ier)
 
See ``fortran/examples/simple1d1.f`` for the complete code,
and below for the complete list of Fortran subroutines available,
and more complicated examples.


.. note ::
 The default demos are double precision. Some single-precision versions have an extra ``f`` after the name, ie, as listed by: ``ls fortran/examples/*f.f``


Summary of Fortran interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The naming of routines is as in C/C++.
Eg, ``finufft2d3`` means 2D transform of type 3.
``finufft2d3many`` means applying 2D transforms of type 3 to a stack of many
strength vectors (vectorized interface).
The guru interface has very similar arguments to its C/C++ version.
Compared to C/C++, all argument lists have ``ier`` appended at the end,
to which the status is written.
Assuming a double-precision build of FINUFFT, these routines and arguments are::
                
       include 'finufft.fh'

       integer ier,iflag,ntrans,type,dim
       integer*8 M,N1,N2,N3,Nk
       integer*8 plan,n_modes(3)
       real*8, allocatable :: xj(:),yj(:),zj(:), sk(:),tk(:),uk(:)
       real*8 tol
       complex*16, allocatable :: cj(:), fk(:)
       type(nufft_opts) opts

 c     simple interface   
       call finufft1d1(M,xj,cj,iflag,tol,N1,fk,opts,ier)
       call finufft1d2(M,xj,cj,iflag,tol,N1,fk,opts,ier)
       call finufft1d3(M,xj,cj,iflag,tol,Nk,sk,fk,opts,ier)
       call finufft2d1(M,xj,yj,cj,iflag,tol,N1,N2,fk,opts,ier)
       call finufft2d2(M,xj,yj,cj,iflag,tol,N1,N2,fk,opts,ier)
       call finufft2d3(M,xj,yj,cj,iflag,tol,Nk,sk,tk,fk,opts,ier)
       call finufft3d1(M,xj,yj,zj,cj,iflag,tol,N1,N2,N3,fk,opts,ier)
       call finufft3d2(M,xj,yj,zj,cj,iflag,tol,N1,N2,N3,fk,opts,ier)
       call finufft3d3(M,xj,yj,zj,cj,iflag,tol,Nk,sk,tk,uk,fk,opts,ier)

 c     vectorized interface
       call finufft1d1many(ntrans,M,xj,cj,iflag,tol,N1,fk,opts,ier)
       call finufft1d2many(ntrans,M,xj,cj,iflag,tol,N1,fk,opts,ier)
       call finufft1d3many(ntrans,M,xj,cj,iflag,tol,Nk,sk,fk,opts,ier)
       call finufft2d1many(ntrans,M,xj,yj,cj,iflag,tol,N1,N2,fk,opts,ier)
       call finufft2d2many(ntrans,M,xj,yj,cj,iflag,tol,N1,N2,fk,opts,ier)
       call finufft2d3many(ntrans,M,xj,yj,cj,iflag,tol,Nk,sk,tk,fk,opts,ier)
       call finufft3d1many(ntrans,M,xj,yj,zj,cj,iflag,tol,N1,N2,N3,fk,opts,ier)
       call finufft3d2many(ntrans,M,xj,yj,zj,cj,iflag,tol,N1,N2,N3,fk,opts,ier)
       call finufft3d3many(ntrans,M,xj,yj,zj,cj,iflag,tol,Nk,sk,tk,uk,fk,opts,ier)

 c     guru interface
       call finufft_makeplan(type,dim,n_modes,iflag,ntrans,tol,plan,opts,ier)
       call finufft_setpts(plan,M,xj,yj,zj,Nk,sk,yk,uk,ier)
       call finufft_exec(plan,cj,fk,ier)
       call finufft_destroy(plan,ier)




       Examples of calling the basic nine routines from fortran are in ``fortran/examples/nufft?d_demo.f`` (for double-precision) and ``fortran/examples/nufft?d_demof.f`` (single-precision). ``fortran/examples/nufft2dmany_demo.f`` shows how to use the vectorized interface.
Here are the calling commands with fortran types for the default double-precision case (the single-precision case is analogous) ::

