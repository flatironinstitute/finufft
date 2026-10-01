c     Simplest fortran example of doing a 1D type 1 transform with FINUFFT,
c     a math test of one output, and how to change from default options.
c     Double-precision (see simple1d1f.f for single).
c     Legacy-style: f77, plus dynamic allocation & derived types from f90.

c     To compile (linux/GCC) from this directory, use eg (paste to one line):

c     gfortran -fopenmp -I../../include simple1d1.f -o simple1d1
c     ../../lib/libfinufft.so -lfftw3 -lfftw3_omp -lgomp -lstdc++

c     Alex Barnett and Libin Lu 5/28/20, fix ptrs 10/6/21
c     (docs-start/docs-end markers let docs/fortran.rst include parts of this
c     tested file; everything outside the math check is shown in the docs)

      program simple1d1
      implicit none

c     docs-start: quick-start
c     our fortran-header, always needed
      include 'finufft.fh'

c     note some inputs are int (int*4) but others BIGINT (int*8)
      integer ier,iflag
      integer*8 N,ktest,M,j,k,ktestindex,t1,t2,crate
      real*8, allocatable :: xj(:)
      real*8 err,tol,pi,fmax,t
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*16, allocatable :: cj(:),fk(:)
      complex*16 fktest

c     this is how you create the options struct in fortran...
      type(finufft_opts) opts
c     or this is if you want default opts, make a null pointer...
      type(finufft_opts), pointer :: defopts => null()
c     docs-end: quick-start

c     how many nonuniform pts
      M = 2000000
c     how many modes
      N = 1000000

      allocate(fk(N))
      allocate(xj(M))
      allocate(cj(M))
      print *,''
      print *,'creating data then run simple interface, default opts...'
c     create some quasi-random NU pts in [-pi, pi), complex strengths
      do j = 1,M
         xj(j) = pi * dcos(pi*j/M)
         cj(j) = dcmplx( dsin((100d0*j)/M), dcos(1.0+(50d0*j)/M))
      enddo

      call system_clock(t1)
c     docs-start: default-call
c     mandatory parameters to FINUFFT: sign of +-i in NUFFT
      iflag = 1
c     tolerance: request 9-digit accuracy
      tol = 1d-9
c     Do transform: writes to fk (mode coeffs), and ier (status flag).
c     use default options:
      call finufft1d1(M,xj,cj,iflag,tol,N,fk,defopts,ier)
c     docs-end: default-call
      call system_clock(t2,crate)
      t = (t2-t1)/float(crate)
      if (ier.eq.0) then
         print '("done in ",f6.3," sec, ",e10.2," NU pts/s")',t,M/t
      else
         print *,'failed! ier=',ier
         stop 1
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
         if (.not.(cdabs(fk(k)).le.huge(fmax))) stop 1
         fmax = max(fmax,cdabs(fk(k)))
      enddo
      ktestindex = ktest + N/2 + 1
      print '("rel err for mode k=",i10," is ",e10.2)',ktest,
     $     cdabs(fk(ktestindex)-fktest)/fmax
      err = cdabs(fk(ktestindex)-fktest)/fmax
      if (.not.(err.le.10*tol)) stop 1

c     docs-start: options
c     do another transform, but now first setting some options...
      print *,''
      print *, 'setting new options, rerun simple interface...'
      call finufft_default_opts(opts)
c     fields of derived type opts may be queried/set as usual...
      opts%debug = 2
      opts%upsampfac = 1.25d0
      print *,'first list our new set of opts vals (cf finufft_opts.h):'
      print *,opts
      call system_clock(t1)
      call finufft1d1(M,xj,cj,iflag,tol,N,fk,opts,ier)
      call system_clock(t2,crate)
      t = (t2-t1)/float(crate)
      if (ier.eq.0) then
         print '("done in ",f6.3," sec, ",e10.2," NU pts/s")',t,M/t
      else
         print *,'failed! ier=',ier
         stop 1
      endif
c     docs-end: options

c     math test of the options-changing transform (same reference & threshold)
      fmax = 0
      do k=1,N
         if (.not.(cdabs(fk(k)).le.huge(fmax))) stop 1
         fmax = max(fmax,cdabs(fk(k)))
      enddo
      print '("rel err for mode k=",i10," is ",e10.2)',ktest,
     $     cdabs(fk(ktestindex)-fktest)/fmax
      err = cdabs(fk(ktestindex)-fktest)/fmax
      if (.not.(err.le.10*tol)) stop 1

      stop
      end
