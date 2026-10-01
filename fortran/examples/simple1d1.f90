! Simplest fortran-90 example of doing a 1D type 1 transform with FINUFFT,
! a math test of one output, and how to change from default options.
! Module version, showing use of finufft_mod.f90
! Double-precision only
! To build and run it, see docs/fortran.rst.

! Alex Barnett, to demo Reinhard Neder f90 module, 1/20/23.

program simple1d1

  ! use the module (which happens to live in ../../include)
  use finufft_mod
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite

  implicit none

  ! fmax is the max abs of the FINUFFT output, used only by the accuracy check
  real*8 fmax

  ! note some inputs are int (int*4) but others BIGINT (int*8)
  integer ier,iflag
  integer*8 N,ktest,M,j,k,ktestindex
  real*8, allocatable :: xj(:)
  real*8 err,tol,pi
  parameter (pi=3.141592653589793238462643383279502884197d0)
  complex*16, allocatable :: cj(:),fk(:)
  complex*16 fktest

  ! this is how you create the options struct in fortran...
  type(finufft_opts) opts
  ! or this is if you want default opts, make a null pointer...
  type(finufft_opts), pointer :: defopts => null()

! docs-start: simple1d1-f90-setup
  ! how many nonuniform pts
  M = 2000000
  ! how many modes
  N = 1000000

  allocate(fk(N))
  allocate(xj(M))
  allocate(cj(M))
  ! create some quasi-random NU pts in [-pi, pi), complex strengths
  do j = 1,M
     xj(j) = pi * dcos(pi*j/M)
     cj(j) = dcmplx( dsin((100d0*j)/M), dcos(1.0+(50d0*j)/M))
  enddo

  ! mandatory parameters to FINUFFT: sign of +-i in NUFFT
  iflag = 1
  ! tolerance
  tol = 1d-9
  ! Do transform: writes to fk (mode coeffs), and ier (status flag).
  ! use default options:
  call finufft1d1(M,xj,cj,iflag,tol,N,fk,defopts,ier)
  ! docs-end: simple1d1-f90-setup
  if (ier.ne.0) then
     print *, 'FAILED: finufft1d1 ier is not 0'
     stop 1, quiet=.true.
  endif

  ! math test: single output mode with given freq (not array index) k
  ktest = N/3
  fktest = dcmplx(0,0)
  do j=1,M
     fktest = fktest + cj(j) * dcmplx( dcos(ktest*xj(j)), &
          dsin(iflag*ktest*xj(j)) )
  enddo
  ! compute inf norm of fk coeffs for use in rel err
  fmax = 0
  do k=1,N
     fmax = max(fmax,cdabs(fk(k)))
  enddo
  ! max() skips NaN, so check the sum over all outputs
  ktestindex = ktest + N/2 + 1
  err = cdabs(fk(ktestindex)-fktest)/fmax
  ! (written so that a NaN err also fails)
  if (.not.ieee_is_finite(sum(abs(fk))) .or. &
       .not.(err.le.10*tol)) then
     print *, 'FAILED: rel err too large, or NaN or Inf in output'
     stop 1, quiet=.true.
  endif

  ! docs-start: simple1d1-f90-options
  ! do another transform, but now first setting some options...
  call finufft_default_opts(opts)
  ! fields of derived type opts may be queried/set as usual...
  opts%upsampfac = 1.25d0
  call finufft1d1(M,xj,cj,iflag,tol,N,fk,opts,ier)
  ! docs-end: simple1d1-f90-options
  if (ier.ne.0) then
     print *, 'FAILED: finufft1d1 ier is not 0'
     stop 1, quiet=.true.
  endif
  fmax = 0
  do k=1,N
     fmax = max(fmax,cdabs(fk(k)))
  enddo
  err = cdabs(fk(ktestindex)-fktest)/fmax
  if (.not.ieee_is_finite(sum(abs(fk))) .or. &
       .not.(err.le.10*tol)) then
     print *, 'FAILED: rel err too large, or NaN or Inf in output'
     stop 1, quiet=.true.
  endif
  print '("rel err = ",e10.2)',err
end program simple1d1
