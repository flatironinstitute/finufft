! Simplest fortran-90 example of doing a 1D type 1 transform with FINUFFT,
! a math test of one output, and how to change from default options.
! Module version, showing use of finufft_mod.f90
! Double-precision only

! To compile (linux/GCC) from this directory, note the module also has to be
! compiled, eg:
      
! gfortran -fopenmp ../../include/finufft_mod.f90 simple1d1.f90 -o simple1d1_f90 ../../lib/libfinufft.so -lfftw3 -lfftw3_omp -lgomp -lstdc++

! Alex Barnett, to demo Reinhard Neder f90 module, 1/20/23.

program simple1d1
  
  ! use the module (which happens to live in ../../include)
  use finufft_mod
  
  implicit none
  
  ! note some inputs are int (int*4) but others BIGINT (int*8)
  integer ier,iflag
  integer*8 N,ktest,M,j,k,ktestindex,t1,t2,crate
  real*8, allocatable :: xj(:)
  real*8 err,tol,pi,fmax,t
  parameter (pi=3.141592653589793238462643383279502884197d0)
  complex*16, allocatable :: cj(:),fk(:)
  complex*16 fktest
  
  ! this is how you create the options struct in fortran...
  type(finufft_opts) opts
  ! or this is if you want default opts, make a null pointer...
  type(finufft_opts), pointer :: defopts => null()
  
  ! how many nonuniform pts
  M = 2000000
  ! how many modes
  N = 1000000

  allocate(fk(N))
  allocate(xj(M))
  allocate(cj(M))
  print *,''
  print *,'creating data then run simple interface, default opts...'
  ! create some quasi-random NU pts in [-pi, pi), complex strengths
  do j = 1,M
     xj(j) = pi * dcos(pi*j/M)
     cj(j) = dcmplx( dsin((100d0*j)/M), dcos(1.0+(50d0*j)/M))
  enddo
  
  call system_clock(t1)
  ! mandatory parameters to FINUFFT: sign of +-i in NUFFT
  iflag = 1
  ! tolerance
  tol = 1d-9
  ! Do transform: writes to fk (mode coeffs), and ier (status flag).
  ! use default options:
  call finufft1d1(M,xj,cj,iflag,tol,N,fk,defopts,ier)
  call system_clock(t2,crate)
  t = (t2-t1)/float(crate)
  if (ier.eq.0) then
     print '("done in ",f6.3," sec, ",e10.2" NU pts/s")',t,M/t
  else
     print *,'failed! ier=',ier
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
  ktestindex = ktest + N/2 + 1
  print '("rel err for mode k=",i10," is ",e10.2)',ktest, &
       cdabs(fk(ktestindex)-fktest)/fmax
      
  ! do another transform, but now first setting some options...
  print *,''
  print *, 'setting new options, rerun simple interface...'
  call finufft_default_opts(opts)
  ! fields of derived type opts may be queried/set as usual...
  opts%debug = 2
  opts%upsampfac = 1.25d0
  print *,'first list our new set of opts vals (cf finufft_opts.h):'
  print *,opts
  call system_clock(t1)
  call finufft1d1(M,xj,cj,iflag,tol,N,fk,opts,ier)
  call system_clock(t2,crate)
  t = (t2-t1)/float(crate)
  if (ier.eq.0) then
     print '("done in ",f6.3," sec, ",e10.2" NU pts/s")',t,M/t
  else
     print *,'failed! ier=',ier
  endif
  
  stop
end program simple1d1
