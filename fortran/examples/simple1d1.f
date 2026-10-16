c     Simplest fortran example of doing a 1D type 1 transform with FINUFFT,
c     a math test of one output, and how to change from default options.
c     Double-precision (see simple1d1f.f for single).
c     Legacy-style: f77, plus dynamic allocation & derived types from f90.
c     To build and run it, see docs/fortran.rst.

c     Alex Barnett and Libin Lu 5/28/20, fix ptrs 10/6/21

      program simple1d1
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      implicit none

c     our fortran-header, always needed
      include 'finufft.fh'

c     note some inputs are int (int*4) but others BIGINT (int*8)
      integer ier,iflag
      integer*8 N,ktest,M,j,k,ktestindex
      real*8, allocatable :: xj(:)
      real*8 err,tol,pi,fmax
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*16, allocatable :: cj(:),fk(:)
      complex*16 fktest

c     this is how you create the options struct in fortran...
      type(finufft_opts) opts
c     or this is if you want default opts, make a null pointer...
      type(finufft_opts), pointer :: defopts => null()

c     how many nonuniform pts
      M = 2000000
c     how many modes
      N = 1000000

      allocate(fk(N))
      allocate(xj(M))
      allocate(cj(M))
c     create some quasi-random NU pts in [-pi, pi), complex strengths
      do j = 1,M
         xj(j) = pi * dcos(pi*j/M)
         cj(j) = dcmplx( dsin((100d0*j)/M), dcos(1.0+(50d0*j)/M))
      enddo

c     mandatory parameters to FINUFFT: sign of +-i in NUFFT
      iflag = 1
c     tolerance
      tol = 1d-9
c     Do transform: writes to fk (mode coeffs), and ier (status flag).
c     use default options:
      call finufft1d1(M,xj,cj,iflag,tol,N,fk,defopts,ier)
      if (ier.ne.0) then
         print *, 'FAILED: ier is not 0'
         stop 1, quiet=.true.
      endif

c     math test: single output mode with given freq (not array index) k
      ktest = N/3
      fktest = dcmplx(0,0)
      do j=1,M
         fktest = fktest + cj(j) * dcmplx( dcos(ktest*xj(j)),
     $        dsin(iflag*ktest*xj(j)) )
      enddo
c     compute inf norm of fk coeffs for use in rel err
      fmax = 0
      do k=1,N
         fmax = max(fmax,cdabs(fk(k)))
      enddo
      ktestindex = ktest + N/2 + 1
      err = cdabs(fk(ktestindex)-fktest)/fmax
c     (written so that a NaN err also fails)
c     max() skips NaN, so check the sum over all outputs
      if (.not.ieee_is_finite(sum(abs(fk))) .or.
     $     .not.(err.le.10*tol)) then
         print *, 'FAILED: rel err too large, or NaN or Inf in output'
         stop 1, quiet=.true.
      endif

c     docs-start: options
c     do another transform, but now first setting some options...
      call finufft_default_opts(opts)
c     fields of derived type opts may be queried/set as usual...
      opts%upsampfac = 1.25d0
      call finufft1d1(M,xj,cj,iflag,tol,N,fk,opts,ier)
c     docs-end: options
      if (ier.ne.0) then
         print *, 'FAILED: ier is not 0'
         stop 1, quiet=.true.
      endif
      fmax = 0
      do k=1,N
         fmax = max(fmax,cdabs(fk(k)))
      enddo
      err = cdabs(fk(ktestindex)-fktest)/fmax
      if (.not.ieee_is_finite(sum(abs(fk))) .or.
     $     .not.(err.le.10*tol)) then
         print *, 'FAILED: rel err too large, or NaN or Inf in output'
         stop 1, quiet=.true.
      endif
      print '("rel err = ",e10.2)',err
      end
