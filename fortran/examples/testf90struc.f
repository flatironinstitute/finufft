c     Attempt to access nufft_opts C struct in f90 struc way
c     Barnett 5/28/20
c
c     Compile with:
c     gfortran-9 -fPIC -O3 -funroll-loops -march=native -fcx-limited-range
c     -I ../../include -fopenmp -I fortran -I /usr/include -fopenmp
c     testf90struc.f
c     ../../lib-static/libfinufft.a -lfftw3 -lm -lgomp -lfftw3_omp -lstdc++
c     -o testf90struc

      program testf77struc
      implicit none
      
c     recreate nufft_opts in f90
      type no
         integer debug, spread_debug,spread_sort,spread_kerevalmeth,
     $        spread_kerpad,chkbnds,fftw,modeord
         real*8 upsampfac
         integer spread_thread,maxbatchsize 
      end type
c     make o an instance of the struct
      type(no) o

c     stuff from example1d1
      integer ier,iflag
      integer*8 N,ktest,M,j,k,ktestindex
      real*8, allocatable :: xj(:)
      real*8 err,tol,pi,t,fmax
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*16, allocatable :: cj(:),fk(:)
      complex*16 fktest

      print *,'testf90struc....'
      call finufft_default_opts(o)
      print *,o%debug,o%spread_sort
c     that seemed to work
      o%debug = 1
      print *,o%debug,o%spread_sort
c     ok      
c     this would be made irrelevant, but tests if idea works...
      call set_debug(o,2)
      print *,o%debug,o%spread_sort

c     rest from example1d1....
c     how many nonuniform pts
      M = 2000000
c     how many modes
      N = 1000000

      allocate(fk(N))
      allocate(xj(M))
      allocate(cj(M))
      print *,'creating data then running simple interface...'
c     create some quasi-random NU pts in [-pi,pi], complex strengths
      do j = 1,M
         xj(j) = pi * dcos(pi*j/M)
         cj(j) = dcmplx( dsin((100d0*j)/M), dcos(1.0+(50d0*j)/M))
      enddo
c     mandatory parameters to FINUFFT: sign of +-i in NUFFT
      iflag = 1
c     tolerance
      tol = 1d-9
c     do it: writes to fk (mode coeffs), and ier (status flag)
c     passes the o by reference (ptr)...
      call finufft1d1(M,xj,cj,iflag,tol,N,fk,o,ier)
      print *,'done. ier=',ier
            
      
      stop
      end
      
