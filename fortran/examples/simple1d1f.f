c     Simplest fortran example of doing a 1D type 1 transform with FINUFFT,
c     a math test of one output, and how to change from default options.
c     Single-precision (see simple1d1.f for double).
c     Legacy-style: f77, plus dynamic allocation & derived types from f90.
c     To build and run it, see docs/fortran.rst.

c     Alex Barnett and Libin Lu 5/28/20, single-prec 6/2/20, ptr 10/6/21

      program simple1d1f
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      implicit none

c     our fortran header, always needed...
      include 'finufft.fh'

c     note some inputs are int (int*4) but others BIGINT (int*8)
      integer ier,iflag
      integer*8 N,ktest,M,j,k,ktestindex
      real*4, allocatable :: xj(:)
      real*4 err,tol,pi,fmax
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*8, allocatable :: cj(:),fk(:)
      complex*8 fktest

c     this is how you create the options struct in fortran...
      type(finufft_opts) opts
c     or this is if you want default opts, make a null pointer...
      type(finufft_opts), pointer :: defopts => null()

c     how many nonuniform pts
      M = 200000
c     how many modes (NB if too large lose acc in single prec)
      N = 10000

      allocate(fk(N))
      allocate(xj(M))
      allocate(cj(M))
c     create some quasi-random NU pts in [-pi, pi), complex strengths
      do j = 1,M
         xj(j) = pi * cos(pi*j/M)
         cj(j) = cmplx( sin((100e0*j)/M), cos(1.0+(50e0*j)/M))
      enddo

c     mandatory parameters to FINUFFT: sign of +-i in NUFFT
      iflag = 1
c     tolerance
      tol = 5e-3
c     Do transform: writes to fk (mode coeffs), and ier (status flag).
c     use default options:
      call finufftf1d1(M,xj,cj,iflag,tol,N,fk,defopts,ier)
      if (ier.ne.0) then
         print *, 'FAILED: finufftf1d1 ier is not 0'
         stop 1, quiet=.true.
      endif

c     math test: single output mode with given freq (not array index) k
      ktest = N/3
      fktest = cmplx(0,0)
      do j=1,M
         fktest = fktest + cj(j) * cmplx( cos(ktest*xj(j)),
     $        sin(iflag*ktest*xj(j)) )
      enddo
c     compute inf norm of fk coeffs for use in rel err
      fmax = 0
      do k=1,N
         fmax = max(fmax,cabs(fk(k)))
      enddo
c     max() skips NaN, so check the sum over all outputs
      ktestindex = ktest + N/2 + 1
      err = cabs(fk(ktestindex)-fktest)/fmax
c     (written so that a NaN err also fails)
      if (.not.ieee_is_finite(sum(abs(fk))) .or.
     $     .not.(err.le.10*tol)) then
         print *, 'FAILED: rel err too large, or NaN or Inf in output'
         stop 1, quiet=.true.
      endif

c     do another transform, but now first setting some options...
      call finufftf_default_opts(opts)
c     fields of derived type opts may be queried/set as usual...
c     note upsampfac is real*8 regardless of the transform precision...
      opts%upsampfac = 1.25d0
c     tell it to ignore that the error model says not possible...
      opts%allow_eps_too_small = 1
      call finufftf1d1(M,xj,cj,iflag,tol,N,fk,opts,ier)
      if (ier.ne.0) then
         print *, 'FAILED: finufftf1d1 ier is not 0'
         stop 1, quiet=.true.
      endif

      fmax = 0
      do k=1,N
         fmax = max(fmax,cabs(fk(k)))
      enddo
      err = cabs(fk(ktestindex)-fktest)/fmax
      if (.not.ieee_is_finite(sum(abs(fk))) .or.
     $     .not.(err.le.10*tol)) then
         print *, 'FAILED: rel err too large, or NaN or Inf in output'
         stop 1, quiet=.true.
      endif
      print '("rel err = ",e10.2)',err
      end
