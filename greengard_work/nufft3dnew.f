c**********************************************************************
      subroutine finufft3d1(nj,xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk,ier)
      implicit none
      integer nj,iflag,ms,mt,mu,ier
      integer k1,k2,k3,next235,nf1,nf2,nf3,n1,itype
      real*8 eps
      real*8 xj(nj),yj(nj),zj(nj),xker,yker,zker
      real*8 params(4)
      complex*16 cj(nj),fk(-ms/2:(ms-1)/2,-mt/2:(mt-1)/2,-mu/2:(mu-1)/2)
      complex *16 cker
c ----------------------------------------------------------------------
      complex*16, allocatable :: fw(:,:,:)
      complex*16, allocatable :: fwker(:,:,:)
      complex*16, allocatable :: fwsav1(:)
      complex*16, allocatable :: fwsav2(:)
      complex*16, allocatable :: fwsav3(:)
c ----------------------------------------------------------------------
c
c     if (iflag .ge. 0) then
c
c                  1  nj
c     fk(k1,k2,k3) = -- SUM cj(j) exp(+i (k1,k2,k3)*(xj(j),yj(j),zj(j)))   
c                 nj j=1 
c                                          for -ms/2 <= k1 <= (ms-1)/2
c                                          for -mt/2 <= k1 <= (mt-1)/2
c                                          for -mu/2 <= k1 <= (mu-1)/2
c     else
c
c                  1  nj
c     fk(k1,k2,k3) = -- SUM cj(j) exp(-i (k1,k2,k3)*(xj(j),yj(j),zj(j)))   
c                 nj j=1 
c                                          for -ms/2 <= k1 <= (ms-1)/2
c                                          for -mt/2 <= k1 <= (mt-1)/2
c                                          for -mu/2 <= k1 <= (mu-1)/2
c
c    References:
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
c     xj,yj,zj  location of sources on [-pi,pi]^2.
c     cj     strengths of sources (complex *16)
c     iflag  determines sign of FFT (see above)
c     eps    precision request 
c     ms     number of Fourier modes computed (-ms/2 to (ms-1)/2 )
c     mt     number of Fourier modes computed (-mt/2 to (mt-1)/2 )
c     mu     number of Fourier modes computed (-mu/2 to (mu-1)/2 )
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
      nf3 = 2*mu
c
      if (2*params(2).gt.nf1) then
         nf1 = next235(2d0*params(2)) 
      endif 
      if (2*params(2).gt.nf2) then
         nf2 = next235(2d0*params(2)) 
      endif 
c
c     allocate arrays for FFTs and initalize FFTs
      allocate(fw(0:nf1-1,0:nf2-1,0:nf3-1))
      allocate(fwker(0:nf1-1,0:nf2-1,0:nf3-1))
c     workspace and init for fftpack:
      allocate(fwsav1(4*nf1+15))
      allocate(fwsav2(4*nf2+15))
      allocate(fwsav3(4*nf3+15))
      call dcffti(nf1,fwsav1)
      call dcffti(nf2,fwsav2)
      call dcffti(nf3,fwsav3)
c
      itype = 1
      n1 = 1
      xker = 0.0d0
      yker = 0.0d0
      zker = 0.0d0
      cker = 1.0d0
c
c     ---------------------------------------------------------------
c     Step 0: get FFT of spreading kernel
c     ---------------------------------------------------------------
c
      call tempspread3d(nf1,nf2,nf3,fwker,n1,xker,yker,zker,cker,
     1                  itype,params)
c
c     ---------------------------------------------------------------
c     Step 1: spread from irregular points to regular grid
c     ---------------------------------------------------------------
c
      call tempspread3d(nf1,nf2,nf3,fw,nj,xj,yj,zj,cj,itype,params)
c
c     ---------------------------------------------------------------
c     Step 2:  Call FFT 
c     ---------------------------------------------------------------
c
      call dcfft3d(iflag,nf1,nf2,nf3,fw,fwsav1,fwsav2,fwsav3)
      call dcfft3d(iflag,nf1,nf2,nf3,fwker,fwsav1,fwsav2,fwsav3)
c
c     ---------------------------------------------------------------
c     Step 3: Deconvolve
c     ---------------------------------------------------------------
c
      do k1 = 0, (ms-1)/2
      do k2 = 0, (mt-1)/2
      do k3 = 0, (mu-1)/2
         fk(k1,k2,k3) = fw(k1,k2,k3)/fwker(k1,k2,k3)/nj
         fk(-k1-1,k2,k3)=fw(nf1-k1-1,k2,k3)/fwker(nf1-k1-1,k2,k3)/nj
         fk(k1,-k2-1,k3)=fw(k1,nf2-k2-1,k3)/fwker(k1,nf2-k2-1,k3)/nj
         fk(-k1-1,-k2-1,k3) = 
     1       fw(nf1-k1-1,nf2-k2-1,k3)/fwker(nf1-k1-1,nf2-k2-1,k3)/nj
c
         fk(k1,k2,-k3-1)=fw(k1,k2,nf3-k3-1)/fwker(k1,k2,nf3-k3-1)/nj
         fk(-k1-1,k2,-k3-1)=
     1       fw(nf1-k1-1,k2,nf3-k3-1)/fwker(nf1-k1-1,k2,nf3-k3-1)/nj
         fk(k1,-k2-1,-k3-1)=
     1       fw(k1,nf2-k2-1,nf3-k3-1)/fwker(k1,nf2-k2-1,nf3-k3-1)/nj
         fk(-k1-1,-k2-1,-k3-1) = fw(nf1-k1-1,nf2-k2-1,nf3-k3-1)
     1       /fwker(nf1-k1-1,nf2-k2-1,nf3-k3-1)/nj
      enddo
      enddo
      enddo
c
      return
      end
c
c**********************************************************************
      subroutine finufft3d2(nj,xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk,ier)
      implicit none
      integer nj,iflag,ms,mt,mu,ier
      integer k1,k2,k3,next235,nf1,nf2,nf3,n1,itype
      real*8 eps
      real*8 xj(nj),yj(nj),zj(nj),xker,yker,zker
      real*8 params(4)
      complex*16 cj(nj),fk(-ms/2:(ms-1)/2,-mt/2:(mt-1)/2,-mu/2:(mu-1)/2)
      complex *16 cker
c ----------------------------------------------------------------------
      complex*16, allocatable :: fw(:,:,:)
      complex*16, allocatable :: fwker(:,:,:)
      complex*16, allocatable :: fwsav1(:)
      complex*16, allocatable :: fwsav2(:)
      complex*16, allocatable :: fwsav3(:)
c ----------------------------------------------------------------------
c
c     if (iflag .ge. 0) then
c
c               (ms-1)/2 (mt-1)/2 (mu-1)/2 
c         cj(j) = SUM    SUM      SUM       fk(k1,k2,k3) * 
c                                 exp(+i (k1,k2,k3)*(xj(j),yj(j),zj(j)))
c              k1=-ms/2 k2=-mt/2 k3=-mu/2        
c                                              for j = 1,...,nj
c
c     else
c
c               (ms-1)/2 (mt-1)/2 (mu-1)/2 
c         cj(j) = SUM    SUM      SUM       fk(k1,k2,k3) * 
c                                 exp(-i (k1,k2,k3)*(xj(j),yj(j),zj(j)))
c              k1=-ms/2 k2=-mt/2 k3=-mu/2        
c                                              for j = 1,...,nj
c
c    References:
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
c     xj,yj,zj  location of sources on [-pi,pi]^2.
c     iflag  determines sign of FFT (see above)
c     eps    precision request 
c     ms     number of Fourier modes computed (-ms/2 to (ms-1)/2 )
c     mt     number of Fourier modes computed (-mt/2 to (mt-1)/2 )
c     mu     number of Fourier modes computed (-mu/2 to (mu-1)/2 )
c     fk     Fourier coefficient values (2D complex *16 array)
c
c     OUTPUT:
c
c     cj     output values (complex *16 array)
c     ier    error return code
c   
c     The type 2 NUFFT proceeds in three steps (see [GL]).
c
c     1) deconvolve (amplify) each Fourier mode first 
c     2) compute inverse FFT on uniform fine grid
c     3) spread data to regular mesh 
c
c ----------------------------------------------------------------------
c     get spreading parameters based on requested precision
      ier = 0
      call get_kernel_params_for_eps_f(params,eps)
c
c     choose DFT size. *** convergence param fix later
      nf1 = 2*ms
      nf2 = 2*mt
      nf3 = 2*mu
      if (2*params(2).gt.nf1) then
         nf1 = next235(2d0*params(2)) 
      endif 
      if (2*params(2).gt.nf2) then
         nf2 = next235(2d0*params(2)) 
      endif 
c
c     allocate arrays for FFTs and initalize FFTs
      allocate(fw(0:nf1-1,0:nf2-1,0:nf3-1))
      allocate(fwker(0:nf1-1,0:nf2-1,0:nf3-1))
c     workspace and init for fftpack:
      allocate(fwsav1(4*nf1+15))
      allocate(fwsav2(4*nf2+15))
      allocate(fwsav3(4*nf3+15))
      call dcffti(nf1,fwsav1)
      call dcffti(nf2,fwsav2)
      call dcffti(nf3,fwsav3)
c
      itype = 1
      n1 = 1
      xker = 0.0d0
      yker = 0.0d0
      zker = 0.0d0
      cker = 1.0d0
c
c     ---------------------------------------------------------------
c     Step 0:  get FFT of spreading kernel
c     ---------------------------------------------------------------
c
      call tempspread3d(nf1,nf2,nf3,fwker,n1,xker,yker,zker,cker,
     1                  itype,params)
      call dcfft3d(iflag,nf1,nf2,nf3,fwker,fwsav1,fwsav2,fwsav3)
c
c
c     ---------------------------------------------------------------
c     Step 1: Deconvolve
c     ---------------------------------------------------------------
c
      do k1 = 0, nf1-1
      do k2 = 0, nf2-1
      do k3 = 0, nf3-1
         fw(k1,k2,k3) = 0.0d0
      enddo
      enddo
      enddo

      do k1 = 0, (ms-1)/2
      do k2 = 0, (mt-1)/2
      do k3 = 0, (mu-1)/2
         fw(k1,k2,k3) = fk(k1,k2,k3)/fwker(k1,k2,k3)
         fw(nf1-k1-1,k2,k3) = fk(-k1-1,k2,k3)/fwker(nf1-k1-1,k2,k3)
         fw(k1,nf2-k2-1,k3) = fk(k1,-k2-1,k3)/fwker(k1,nf2-k2-1,k3)
         fw(nf1-k1-1,nf2-k2-1,k3) =
     1       fk(-k1-1,-k2-1,k3)/fwker(nf1-k1-1,nf2-k2-1,k3)
c
         fw(k1,k2,nf3-k3-1) = fk(k1,k2,-k3-1)/fwker(k1,k2,nf3-k3-1)
         fw(nf1-k1-1,k2,nf3-k3-1) =
     1    fk(-k1-1,k2,-k3-1)/fwker(nf1-k1-1,k2,nf3-k3-1)
         fw(k1,nf2-k2-1,nf3-k3-1) =
     1    fk(k1,-k2-1,-k3-1)/fwker(k1,nf2-k2-1,nf3-k3-1)
         fw(nf1-k1-1,nf2-k2-1,nf3-k3-1) = fk(-k1-1,-k2-1,-k3-1)
     1       /fwker(nf1-k1-1,nf2-k2-1,nf3-k3-1)
      enddo
      enddo
      enddo
c
c     ---------------------------------------------------------------
c     Step 2:  Call FFT 
c     ---------------------------------------------------------------
c
      call dcfft3d(iflag,nf1,nf2,nf3,fw,fwsav1,fwsav2,fwsav3)
c
c     ---------------------------------------------------------------
c     Step 3:  Spread from uniform grid to irregular points
c     ---------------------------------------------------------------
c
      itype = 2
      call tempspread3d(nf1,nf2,nf3,fw,nj,xj,yj,zj,cj,itype,params)
c
      return
      end
c
