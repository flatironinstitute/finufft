c     Simplest fortran example of doing a 1D type 1 transform with FINUFFT,
c     setting various options, and math test of one output.
c     Style is f77 plus dynamic allocation from f90.
c     Double-precision (see example1d1f.f for single).
c
c     To compile (GCC) use, eg:
c     gfortran-9 

c     Alex Barnett and Libin Lu 5/28/20
      program example1d1
      implicit none
c     this purely for wall-clock timer...
      include 'omp_lib.h'

c     note some inputs are int (int*4) but others BIGINT (int*8)
      integer ier,iflag
      integer*8 N,ktest,M,j,k,ktestindex
      real*8, allocatable :: xj(:)
      real*8 err,tol,pi,t,fmax
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*16, allocatable :: cj(:),fk(:)
      complex*16 fktest

c     following (if never allocated) passes a NULL ptr to C...
      integer*8, allocatable :: null
c     any 8-byte holder for a C pointer (to nufft_opts struct)...
      integer*8 opts
      
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

      t = omp_get_wtime()
c     mandatory parameters to FINUFFT: sign of +-i in NUFFT
      iflag = 1
c     tolerance
      tol = 1d-9
c     do it: writes to fk (mode coeffs), and ier (status flag)
c     null here uses default options
      call finufft1d1(M,xj,cj,iflag,tol,N,fk,null,ier)
      t = omp_get_wtime()-t
      print '("done (ier=",i2,"), ",f6.3," sec, ",e10.2" NU pts/s")',
     $     ier,t,M/t
      if (ier.ne.0) stop
      
c     check a single output mode with given freq (not array index) k
      ktest = N/3
      fktest = dcmplx(0,0)
      do j=1,M
         fktest = fktest + cj(j) * dcmplx( dcos(ktest*xj(j)),
     $        dsin(iflag*ktest*xj(j)) )
      enddo
c     print *,(fk(k), k=1,N)
c     compute inf norm of fk coeffs for use in rel err
      fmax = 0
      do k=1,N
         fmax = max(fmax,cdabs(fk(k)))
      enddo
      ktestindex = ktest + N/2 + 1
      print '("rel err for mode k=",i10,":",e10.2)',ktest,
     $     cdabs(fk(ktestindex)-fktest)/fmax
      
      
c     do it again with instead setting some options...
      print *,'setting new options and rerunning simple interface...'
c     opts is a opaque pointer to the nufft_opts struct in C
c     print *,opts
c     *** THIS CURRENTLY FAILS:
      call finufft_default_opts(opts)
c      print *,opts
c     (opts is anything*8, a "blind pointer" to the C++ nufft_opts struct)
      call set_debug(opts,2)
      call set_upsampfac(opts,1.25d0)
c     weirdly the value of opts is changed but set_ which it shouldn't be...
c      print *,opts
      t = omp_get_wtime()
      call finufft1d1(M,xj,cj,iflag,tol,N,fk,opts,ier)
      t = omp_get_wtime()-t
      print '("done (ier=",i2,"), ",f6.3," sec, ",e10.2" NU pts/s")',
     $     ier,t,M/t
      
      stop
      end
