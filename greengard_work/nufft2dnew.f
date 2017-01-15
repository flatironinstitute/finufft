c**********************************************************************
      subroutine finufft2d1(nj,xj,yj,cj,iflag,eps,ms,mt,fk,ier)
      implicit none
      integer nj,iflag,ms,mt,ier
      integer k1,k2,next235,nf1,nf2,n1,itype
      real*8 eps
      real*8 xj(nj),yj(nj),xker,yker
      real*8 params(4)
      complex*16 cj(nj),fk(-ms/2:(ms-1)/2,-mt/2:(mt-1)/2)
      complex*16 cker
c ----------------------------------------------------------------------
      complex*16, allocatable :: fw(:,:)
      complex*16, allocatable :: fwker(:,:)
      complex*16, allocatable :: fwsav1(:)
      complex*16, allocatable :: fwsav2(:)
c ----------------------------------------------------------------------
c     if (iflag .ge. 0) then
c
c               1  nj
c     fk(k1,k2) = -- SUM cj(j) exp(+i (k1,k2) * (xj(j, yj(j)) )   
c               nj j=1                     
c                                   for -ms/2 <= k1 <= (ms-1)/2
c                                   for -mt/2 <= k1 <= (mt-1)/2
c
c    else 
c
c               1  nj
c     fk(k1,k2) = -- SUM cj(j) exp(-i (k1,k2) * (xj(j, yj(j)) )   
c               nj j=1                     
c                                   for -ms/2 <= k1 <= (ms-1)/2
c                                   for -mt/2 <= k1 <= (mt-1)/2
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
c     xj,yj  location of sources on [-pi,pi]^2.
c     cj     strengths of sources (complex *16)
c     iflag  determines sign of FFT (see above)
c     eps    precision request 
c     ms     number of Fourier modes computed (-ms/2 to (ms-1)/2 )
c     mt     number of Fourier modes computed (-mt/2 to (mt-1)/2 )
c
c     OUTPUT:
c
c     fk     Fourier transform values (2D complex *16)
c     ier    error return code
c   
c     The type 1 NUFFT proceeds in three steps (see [GL]).
c
c     1) spread data to oversampled regular mesh
c     2) compute FFT on uniform mesh
c     3) deconvolve each Fourier mode independently
c
c ----------------------------------------------------------------------
c     get spreading parameters based on requested precision
      ier = 0
      call get_kernel_params_for_eps_f(params,eps)
c
c     choose DFT size. *** convergence param fix later
      nf1 = 2*ms
      nf2 = 2*mt
c
      if (2*params(2).gt.nf1) then
         nf1 = next235(2d0*params(2)) 
      endif 
      if (2*params(2).gt.nf2) then
         nf2 = next235(2d0*params(2)) 
      endif 
c
c     allocate arrays for FFTs and initalize FFTs
      allocate(fw(0:nf1-1,0:nf2-1))
      allocate(fwker(0:nf1-1,0:nf2-1))
c     workspace and init for fftpack:
      allocate(fwsav1(4*nf1+15))
      allocate(fwsav2(4*nf2+15))
      call dcffti(nf1,fwsav1)
      call dcffti(nf2,fwsav2)
c
      itype = 1
      n1 = 1
      xker = 0.0d0
      yker = 0.0d0
      cker = 1.0d0
c
c     ---------------------------------------------------------------
c     Step 0: get FFT of spreading kernel
c     ---------------------------------------------------------------
c
      call tempspread2d(nf1,nf2,fwker,n1,xker,yker,cker,itype,params)
c
c     ---------------------------------------------------------------
c     Step 1: spread from irregular points to regular grid
c     ---------------------------------------------------------------
c
      call tempspread2d(nf1,nf2,fw,nj,xj,yj,cj,itype,params)
c
c     ---------------------------------------------------------------
c     Step 2:  Call FFT 
c     ---------------------------------------------------------------
c
      call dcfft2d(iflag,nf1,nf2,fw,fwsav1,fwsav2)
      call dcfft2d(iflag,nf1,nf2,fwker,fwsav1,fwsav2)
c
c     ---------------------------------------------------------------
c     Step 3: Deconvolve
c     ---------------------------------------------------------------
c
      do k1 = 0, (ms-1)/2
      do k2 = 0, (mt-1)/2
         fk(k1,k2) = fw(k1,k2)/fwker(k1,k2)/nj
         fk(-k1-1,k2) = fw(nf1-k1-1,k2)/fwker(nf1-k1-1,k2)/nj
         fk(k1,-k2-1) = fw(k1,nf2-k2-1)/fwker(k1,nf2-k2-1)/nj
         fk(-k1-1,-k2-1) = 
     1       fw(nf1-k1-1,nf2-k2-1)/fwker(nf1-k1-1,nf2-k2-1)/nj
      enddo
      enddo
c
      return
      end
c
c**********************************************************************
      subroutine finufft2d2(nj,xj,yj,cj,iflag,eps,ms,mt,fk,ier)
      implicit none
      integer nj,iflag,ms,mt,ier
      integer k1,k2,next235,nf1,nf2,n1,itype
      real*8 eps
      real*8 xj(nj),yj(nj),xker,yker
      real*8 params(4)
      complex*16 cj(nj),fk(-ms/2:(ms-1)/2,-mt/2:(mt-1)/2),cker
c ----------------------------------------------------------------------
      complex*16, allocatable :: fw(:,:)
      complex*16, allocatable :: fwker(:,:)
      complex*16, allocatable :: fwsav1(:)
      complex*16, allocatable :: fwsav2(:)
c ----------------------------------------------------------------------
c     if (iflag .ge. 0) then
c
c         (ms-1)/2 (mt-1)/2
c    cj(j) = SUM    SUM    fk(k1,k2) exp(+i (k1,k2) * (xj(j),yj(j)) )   
c         k1=-ms/2 k2=-mt/2             
c                                           for j = 1,...,nj
c     else
c
c         (ms-1)/2 (mt-1)/2
c    cj(j) = SUM    SUM    fk(k1,k2) exp(-i (k1,k2) * (xj(j),yj(j)) )   
c         k1=-ms/2 k2=-mt/2             
c                                           for j = 1,...,nj
c
c ----------------------------------------------------------------------
c     INPUT:
c
c     nj     number of output values   (integer)
c     xj,yj  location of output values (real *8 arrays)
c     iflag  determines sign of FFT (see above)
c     eps    precision request  (between 1.0d-33 and 1.0d-1)
c               recomended value is 1d-15 for double precision calculations
c     ms     number of Fourier modes given (1st index)  [-ms/2:(ms-1)/2]
c     mt     number of Fourier modes given (2nd index)  [-mt/2:(mt-1)/2]
c     fk     Fourier coefficient values (complex *16 2D array)
c
c     OUTPUT:
c
c     cj     output values (complex *16 array)
c     ier    error return code
c   
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
c     get spreading parameters based on requested precision
      ier = 0
      call get_kernel_params_for_eps_f(params,eps)
c
c     choose DFT size. *** convergence param fix later
      nf1 = 2*ms
      nf2 = 2*mt
      if (2*params(2).gt.nf1) then
         nf1 = next235(2d0*params(2)) 
      endif 
      if (2*params(2).gt.nf2) then
         nf2 = next235(2d0*params(2)) 
      endif 
c
c     allocate arrays for FFTs and initalize FFTs
      allocate(fw(0:nf1-1,0:nf2-1))
      allocate(fwker(0:nf1-1,0:nf2-1))
c     workspace and init for fftpack:
      allocate(fwsav1(4*nf1+15))
      allocate(fwsav2(4*nf2+15))
      call dcffti(nf1,fwsav1)
      call dcffti(nf2,fwsav2)
c
      itype = 1
      n1 = 1
      xker = 0.0d0
      yker = 0.0d0
      cker = 1.0d0
c
c     ---------------------------------------------------------------
c     Step 0:  get FFT of spreading kernel
c     ---------------------------------------------------------------
c
      call tempspread2d(nf1,nf2,fwker,n1,xker,yker,cker,itype,params)
      call dcfft2d(iflag,nf1,nf2,fwker,fwsav1,fwsav2)
c
c     ---------------------------------------------------------------
c     Step 1: Deconvolve
c     ---------------------------------------------------------------
c
      do k1 = 0, nf1-1
      do k2 = 0, nf2-1
         fw(k1,k2) = 0.0d0
      enddo
      enddo
c
      do k1 = 0, (ms-1)/2
      do k2 = 0, (mt-1)/2
         fw(k1,k2) = fk(k1,k2)/fwker(k1,k2)
         fw(nf1-k1-1,k2) = fk(-k1-1,k2)/fwker(nf1-k1-1,k2)
         fw(k1,nf2-k2-1) = fk(k1,-k2-1)/fwker(k1,nf2-k2-1)
         fw(nf1-k1-1,nf2-k2-1) = 
     1       fk(-k1-1,-k2-1)/fwker(nf1-k1-1,nf2-k2-1)
      enddo
      enddo
c
c     ---------------------------------------------------------------
c     Step 2:  Call FFT 
c     ---------------------------------------------------------------
c
      call dcfft2d(iflag,nf1,nf2,fw,fwsav1,fwsav2)
c
c     ---------------------------------------------------------------
c     Step 3:  Spread from uniform grid to irregular points
c     ---------------------------------------------------------------
c
      itype = 2
      call tempspread2d(nf1,nf2,fw,nj,xj,yj,cj,itype,params)
c
      return
      end
c
