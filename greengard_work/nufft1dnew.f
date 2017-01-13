c**********************************************************************
      subroutine finufft1d1(nj,xj,cj,iflag,eps,ms,fk,ier)
      implicit none
      integer ier,iflag,n1,itype
      integer k1,ms,next235,nf1,nj
      real*8 eps
      real*8 xj(nj),xker
      complex*16 cj(nj),fk(-ms/2:(ms-1)/2),cker
c ----------------------------------------------------------------------
      real*8, allocatable :: params(:)
      complex*16, allocatable :: fw(:)
      complex*16, allocatable :: fwker(:)
      complex*16, allocatable :: fwsav(:)
c ----------------------------------------------------------------------
c     if (iflag .ge. 0) then
c
c               1  nj
c     fk(k1) = -- SUM cj(j) exp(+i k1 xj(j))  for -ms/2 <= k1 <= (ms-1)/2 
c              nj j=1                            
c
c     else
c
c               1  nj
c     fk(k1) = -- SUM cj(j) exp(-i k1 xj(j))  for -ms/2 <= k1 <= (ms-1)/2 
c              nj j=1                            
c
c     References:
c
c     [DR] Fast Fourier transforms for nonequispaced data,
c          A. Dutt and V. Rokhlin, SIAM J. Sci. Comput. 14, 
c          1368-1383, 1993.
c
c     [GL] Accelerating the Nonuniform Fast Fourier Transform,
c          L. Greengard and J.-Y. Lee, SIAM Review 46, 443-454 (2004).
c
c ----------------------------------------------------------------------
c     INPUT:
c
c     nj     number of sources
c     xj     location of sources on interval [-pi,pi].
c     cj     strengths of sources (complex *16)
c     iflag  determines sign of FFT (see above)
c     eps    precision requested
c     ms     number of Fourier modes computed (-ms/2 to (ms-1)/2 )
c
c     OUTPUT:
c
c     fk     Fourier transform values (complex *16)
c     ier    error return code
c            WHAT TO FLAG???
c
c     The type 1 NUFFT proceeds in three steps (see [GL]).
c
c     1) spread data to oversampled regular mesh 
c     2) compute FFT on uniform mesh
c     3) deconvolve each Fourier mode independently
c
c ----------------------------------------------------------------------
c
c     get spreading parameters based on requwsted precision
c
      ier = 0
      allocate(params(4))
      call get_kernel_params_for_eps_f(params,eps)
      nf1 = 2*ms
      if (2*params(2).gt.nf1) then
         nf1 = next235(2d0*params(2)) 
      endif 
c
c     allocate arrays for FFTs and initalize FFTs
c
      allocate(fw(0:nf1-1))
      allocate(fwker(0:nf1-1))
      allocate(fwsav(4*nf1+15))
      call dcffti(nf1,fwsav)
c
      itype = 1
      call tempspread1d(nf1,fw,nj,xj,cj,itype,params)
      n1 = 1
      xker = 0.0d0
      cker = 1.0d0
      call tempspread1d(nf1,fwker,n1,xker,cker,itype,params)
c
c     ---------------------------------------------------------------
c     Call 1D FFT 
c     ---------------------------------------------------------------
c
      if (iflag .ge. 0) then
         call dcfftb(nf1,fw,fwsav)
         call dcfftb(nf1,fwker,fwsav)
      else
         call dcfftf(nf1,fw,fwsav)
         call dcfftf(nf1,fwker,fwsav)
      endif
c
c     ---------------------------------------------------------------
c     Deconvolve
c     ---------------------------------------------------------------
c
      fk(0) = fw(0)/fwker(0)/ms
      do k1 = 1, (ms-1)/2
         fk(k1) = fw(k1)/fwker(k1)/ms
         fk(-k1) = fw(nf1-k1)/fwker(nf1-k1)/ms
      enddo
      if (ms/2*2.eq.ms) then
         fk(-ms/2) = fw(nf1-ms/2)/fwker(nf1-ms/2)/ms
      endif
c
      return
      end
c
c
c
c
c
c
c**********************************************************************
      subroutine finufft1d2(nj,xj,cj,iflag,eps,ms,fk,ier)
      implicit none
      integer ier,iflag,n1,itype
      integer j,k1,ms,next235,nf1,nj
      real*8 eps
      real*8 xj(nj),xker
      real*8 params(4)
      complex*16 cj(nj),fk(-ms/2:(ms-1)/2),cker
c ----------------------------------------------------------------------
      complex*16, allocatable :: fw(:)
      complex*16, allocatable :: fwker(:)
      complex*16, allocatable :: fwsav(:)
c ----------------------------------------------------------------------
c     if (iflag .ge. 0) then
c
c              (ms-1)/2
c     cj(j) =    SUM      fk(k1) exp(+i k1 xj(j))  for j = 1,...,nj
c              k1= -ms/2                            
c
c     else
c
c              (ms-1)/2
c     cj(j) =    SUM      fk(k1) exp(-i k1 xj(j))  for j = 1,...,nj
c              k1= -ms/2                            
c
c ----------------------------------------------------------------------
c     INPUT:
c
c     nj     number of output values   (integer)
c     xj     location of output values (real *8 array)
c     iflag  determines sign of FFT (see above)
c     eps    precision request  (between 1.0d-33 and 1.0d-1)
c               recomended value is 1d-15 for double precision calculations
c     ms     number of Fourier modes given  [ -ms/2: (ms-1)/2 ]
c     fk     Fourier coefficient values (complex *16 array)
c
c     OUTPUT:
c
c     cj     output values (complex *16 array)
c     ier    error return code
c   
c            ier = 0  => normal execution.
c            ier = 1  => precision eps requested is out of range.
c
c     The type 2 algorithm proceeds in three steps (see [GL]).
c
c     1) deconvolve (amplify) each Fourier mode first
c     2) compute inverse FFT on uniform fine grid
c     3) spread data to regular mesh using Gaussian
c ----------------------------------------------------------------------
      ier = 0
      call prini(6,13)
      call get_kernel_params_for_eps_f(params,eps)
      nf1 = 2*ms
      if (2*params(2).gt.nf1) then
         nf1 = next235(2d0*params(2)) 
      endif 
      call prinf(' nf1 is *',nf1,1)
c
      allocate(fw(0:nf1-1))
      allocate(fwker(0:nf1-1))
      allocate(fwsav(4*nf1+15))
      call dcffti(nf1,fwsav)
c
      itype = 1
      n1 = 1
      xker = 0.0d0
      cker = 1.0d0
      call tempspread1d(nf1,fwker,n1,xker,cker,itype,params)
c
      if (iflag .ge. 0) then
         call dcfftb(nf1,fwker,fwsav)
      else
         call dcfftf(nf1,fwker,fwsav)
      endif
c
c     ---------------------------------------------------------------
c     Deconvolve
c     ---------------------------------------------------------------
c
      fw(0) = fk(0)/fwker(0)/ms
      do k1 = 1, (ms-1)/2
         fw(k1) = fk(k1)/fwker(k1)/ms
         fw(nf1-k1) = fk(-k1)/fwker(nf1-k1)/ms
      enddo
      fw(nf1-ms/2) = fk(-ms/2)/fwker(nf1-ms/2)/ms
      do k1 = (ms+1)/2, nf1-ms/2-1
         fw(k1) = dcmplx(0d0, 0d0)
      enddo
c
c     ---------------------------------------------------------------
c     Call 1D FFT 
c     ---------------------------------------------------------------
c
      if (iflag .ge. 0) then
         call dcfftb(nf1,fw,fwsav)
      else
         call dcfftf(nf1,fw,fwsav)
      endif
      call prin2(' fw is *',fw(0),2*nf1)
c
      itype = 2
      call tempspread1d(nf1,fw,nj,xj,cj,itype,params)

      return
      end
c
