.. _fort:

Using FINUFFT from fortran
==========================

We provide fortran interfaces that are very similar to those in C/C++.

Simple example
**************

*** TO DO ....



plus older test codes changed little from the CMCL NUFFT library apart from
that they now call the FINUFFT library.

The interfaces are identical to those of CMCL (ie drop-in replacements),
apart from the type-1 no longer have the 1/nj normalization.
The naming is:
finufftNdM_f(...)  where N=dimensions (1,2 or 3) and M=type (1,2 or 3).

Note that, on a linux system, to compile and
link a Fortran program against the FINUFFT
library, use the following:

gfortran nufft1d_demo.f dirft1d.f -o nufft1d_demo ../lib/libfinufft.a -lstdc++ -lfftw3 -lfftw3_omp -lm -fopenmp

For Mac OSX, replace fftw3_omp by fftw3_threads.
Or, if you compiled a single-threaded version:

gfortran nufft1d_demo.f dirft1d.f -o nufft1d_demo ../lib/libfinufft.a -lstdc++ -lfftw3 -lm

Alternatively you may want to compile with g++ and use -lgfortran at the *end* of the compile statement.

See ../makefile
Eg
(cd ..; make fortran)

Finally, the default demos are double precision. The single-precision
versions have an extra "f" after the name, ie as listed by: ls *f.f






The meaning of arguments is as in the C++ documentation above,
apart from that now ``ier`` is an argument which is output to.
Examples of calling the basic 9 routines from fortran are in ``fortran/nufft?d_demo.f`` (for double-precision) and ``fortran/nufft?d_demof.f`` (single-precision). ``fortran/nufft2dmany_demo.f`` shows how to use the many-vector interface.
Here are the calling commands with fortran types for the default double-precision case (the simple-precision case is analogous) ::

      integer ier,iflag,ms,mt,mu,nj,ndata
      real*8, allocatable :: xj(:),yj(:),zj(:), sk(:),tk(:),uk(:)
      real*8 err,eps
      complex*16, allocatable :: cj(:), fk(:)

      call finufft1d1_f(nj,xj,cj,iflag,eps, ms,fk,ier)
      call finufft1d2_f(nj,xj,cj,iflag, eps, ms,fk,ier)
      call finufft1d3_f(nj,xj,cj,iflag,eps, ms,sk,fk,ier)
      call finufft2d1_f(nj,xj,yj,cj,iflag,eps,ms,mt,fk,ier)
      call finufft2d1many_f(ndata,nj,xj,yj,cj,iflag,eps,ms,mt,fk,ier)
      call finufft2d2_f(nj,xj,yj,cj,iflag,eps,ms,mt,fk,ier)
      call finufft2d2many_f(ndata,nj,xj,yj,cj,iflag,eps,ms,mt,fk,ier)
      call finufft2d3_f(nj,xj,yj,cj,iflag,eps,nk,sk,tk,fk,ier)
      call finufft3d1_f(nj,xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk,ier)
      call finufft3d2_f(nj,xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk,ier)
      call finufft3d3_f(nj,xj,yj,zj,cj,iflag,eps,nk,sk,tk,uk,fk,ier)


