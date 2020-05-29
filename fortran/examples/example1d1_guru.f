c     GURU TEST.
c     setting various options, and math test of one output.
c     Style is f77 plus dynamic allocation from f90.

c     Alex Barnett and Libin Lu 5/28/20
      program example1d1_guru
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
      integer*8 plan,zero
      integer*8, allocatable :: n_modes(:)
      integer type,dim,ntrans

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
      print *,'creating data then running guru interface...'
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
      allocate(n_modes(3))
      n_modes(1) = N
      n_modes(2) = 0
      n_modes(3) = 0
      type=1
      dim=1
      ntrans=1
      zero=0
c     if I now print *,plan ... it causes the rest to corrupt or segfault!
      call finufft_makeplan(type,dim,n_modes,iflag,ntrans,
     $     tol,plan,null,ier)
      call finufft_setpts(plan,M,xj,null,null,zero,
     $           null,null,null,ier)
      call finufft_exec(plan,cj,fk,ier)
      call finufft_destroy(plan,ier)
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
      
      stop
      end
