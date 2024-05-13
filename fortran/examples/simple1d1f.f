c     Simplest fortran example of doing a 1D type 1 transform with FINUFFT,
c     a math test of one output, and how to change from default options.
c     Single-precision (see simple1d1.f for double).
c     Legacy-style: f77, plus dynamic allocation & derived types from f90.

c     To compile (linux/GCC) from this directory, use eg (paste to one line):
      
c     gfortran -fopenmp -I../../include simple1d1f.f -o simple1d1f
c     -L../../lib -lfinufftf

c     Alex Barnett and Libin Lu 5/28/20, single-prec 6/2/20, ptr 10/6/21

      program simple1d1f
      implicit none
      
c     our fortran header, always needed...
      include 'finufft.fh'

c     note some inputs are int (int*4) but others BIGINT (int*8)
      integer ier,iflag
      integer*8 N,ktest,M,j,k,ktestindex,t1,t2,crate
      real*4, allocatable :: xj(:)
      real*4 err,tol,pi,t,fmax
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*8, allocatable :: cj(:),fk(:)
      complex*8 fktest

c     this is how you create the options struct in fortran...
      type(finufft_opts) opts
c     or this is if you want default opts, make a null pointer...
      type(finufft_opts), pointer :: defopts => null()
     
c     how many nonuniform pts
      M = 200000
c     how many modes
      N = 100000

      allocate(fk(N))
      allocate(xj(M))
      allocate(cj(M))
      print *,''
      print *,'creating data then run simple interface, default opts...'
c     create some quasi-random NU pts in [-pi, pi), complex strengths
      do j = 1,M
         xj(j) = pi * cos(pi*j/M)
         cj(j) = cmplx( sin((100e0*j)/M), cos(1.0+(50e0*j)/M))
      enddo

      call system_clock(t1)
c     mandatory parameters to FINUFFT: sign of +-i in NUFFT
      iflag = 1
c     tolerance
      tol = 1e-6
c     Do transform: writes to fk (mode coeffs), and ier (status flag).
c     use default options:
      call finufftf1d1(M,xj,cj,iflag,tol,N,fk,defopts,ier)
      call system_clock(t2,crate)
      t = (t2-t1)/float(crate)
      if (ier.eq.0) then
         print '("done in ",f6.3," sec, ",e10.2" NU pts/s")',t,M/t
      else
         print *,'failed! ier=',ier
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
      ktestindex = ktest + N/2 + 1
      print '("rel err for mode k=",i10," is ",e10.2)',ktest,
     $     cabs(fk(ktestindex)-fktest)/fmax
      
c     do another transform, but now first setting some options...
      print *,''
      print *, 'setting new options, rerun simple interface...'
      call finufftf_default_opts(opts)
c     fields of derived type opts may be queried/set as usual...
      opts%debug = 2
c     note upsampfac is real*8 regardless of the transform precision...
      opts%upsampfac = 1.25d0
      print *,'first list our new set of opts vals (cf finufft_opts.h):'
      print *,opts
      call system_clock(t1)
      call finufftf1d1(M,xj,cj,iflag,tol,N,fk,opts,ier)
      call system_clock(t2,crate)
      t = (t2-t1)/float(crate)
      if (ier.eq.0) then
         print '("done in ",f6.3," sec, ",e10.2" NU pts/s")',t,M/t
      else
         print *,'failed! ier=',ier
      endif
      
      stop
      end
