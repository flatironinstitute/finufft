cccc alex hack at simple driver demo of new interfaces.
      program nufft1d1_demo
      implicit none
c
c --- local variables
c
      integer i,ier,iflag,j,k1,mx
      integer*8 ms,mstest,nj
      real*8, allocatable :: xj(:),sk(:)
      real*8 err,eps,pi
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*16, allocatable :: cj(:),fk0(:),fk1(:)
c     if never allocated, this makes a NULL ptr when passed to C++...
      integer*8, allocatable :: null
c     this will be a "blind pointer" to the C++ nufft_opts struct...
      integer*8 opt
c
c     --------------------------------------------------
c     create some test data
c     --------------------------------------------------
      ms = 1000000
      nj = 2000000
c     modes to check
      mstest = 10
c     first alloc everything
      allocate(fk1(ms))
      allocate(fk0(mstest))
      allocate(sk(ms))
      allocate(xj(nj))
      allocate(cj(nj))
      do k1 = -nj/2, (nj-1)/2
         j = k1+nj/2+1
         xj(j) = pi * dcos(-pi*j/nj)
         cj(j) = dcmplx( dsin(pi*j/nj), dcos(pi*j/nj))
      enddo
c
c     --------------------------------------------------
c     start tests
c     --------------------------------------------------
c
      iflag = 1
      print*,' Start 1D testing: ', ' nj =',nj, ' ms =',ms
      eps = 1d-6
c     -----------------------
c     call 1D Type1 method
c     -----------------------
c     only check a small # central modes...
      call dirft1d1(nj,xj,cj,iflag, mstest,fk0)
      
      call finufft1d1(nj,xj,cj,iflag,eps, ms,fk1,null,ier)

c     check only the central mstest modes... (ms, mstest must be even!)
      call errcomp(fk0,fk1((ms-mstest)/2),mstest,err)

      print *,' ier = ',ier
c     *** still need to debug alignment of modes for check! make simpler.
      print *,' type 1 error = ',err

c     now instead try options setting...
      call finufft_default_opts(opt)
      call set_debug(opt,2)
      call finufft1d1(nj,xj,cj,iflag,eps, ms,fk1,opt,ier)
      
      stop
      end
c
c
c
c
c
      subroutine errcomp(fk0,fk1,n,err)
      implicit none
      integer*8 k,n
      complex*16 fk0(n), fk1(n)
      real *8 salg,ealg,err
c
      ealg = 0d0
      salg = 0d0
      do k = 1, n
         ealg = ealg + cdabs(fk1(k)-fk0(k))**2
         salg = salg + cdabs(fk0(k))**2
      enddo
      err =sqrt(ealg/salg)
      return
      end
