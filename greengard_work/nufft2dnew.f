c**********************************************************************
      subroutine nufft2d1f90x(nj,xj,yj,cj,iflag,eps,ms,mt,fk,ier)
      implicit none
      integer ier,iflag,istart,iw1,iwtot,iwsav,n1,itype
      integer j,jb1,jb1u,jb1d,k1,k2,ms,mt,next235,nf1,nf2,nj,nspread
      real*8 cross,cross1,diff1,eps,hx,pi,rat,r2lamb,t1,tau
      real*8 xj(nj),yj(nj),xker,yker,rfac
      real*8 params(4)
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*16 cj(nj),fk(-ms/2:(ms-1)/2,-mt/2:(mt-1)/2),zz,ccj,cker
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
      ier = 0
      call prini(6,13)
      call get_kernel_params_for_eps_f(params,eps)
      nf1 = 2*ms
      nf2 = 2*mt
ccc      if (2*params(2).gt.nf1) then
ccc         nf1 = next235(2d0*params(2)) 
ccc      endif 
ccc      if (2*params(2).gt.nf2) then
ccc         nf2 = next235(2d0*params(2)) 
ccc      endif 
      call prinf(' nf1 is *',nf1,1)
      call prinf(' nf2 is *',nf2,1)
c
      allocate(fw(0:nf1-1,0:nf2-1))
      allocate(fwker(0:nf1-1,0:nf2-1))
      allocate(fwsav1(4*nf1+15))
      allocate(fwsav2(4*nf2+15))
      call dcffti(nf1,fwsav1)
      call dcffti(nf2,fwsav2)
c
      itype = 1
      call tempspread2d(nf1,nf2,fw,nj,xj,yj,cj,itype,params)
      write(6,*) fw(0,0), fw(0,1), fw(1,0), fw(1,1)
      n1 = 1
      xker = 0.0d0
      yker = 0.0d0
      cker = 1.0d0
      call tempspread2d(nf1,nf2,fwker,n1,xker,yker,cker,itype,params)
      write(6,*) fwker(0,0), fwker(0,1), fwker(1,0), fwker(1,1)
c
c     ---------------------------------------------------------------
c     Call 2D FFT 
c     ---------------------------------------------------------------
c
      call dcfft2d(iflag,nf1,nf2,fw,fwsav1,fwsav2)
      call dcfft2d(iflag,nf1,nf2,fwker,fwsav1,fwsav2)
      write(6,*) 'after', fw(0,0), fw(0,1), fw(1,0), fw(1,1)
      write(6,*) 'after', fwker(0,0), fwker(0,1), fwker(1,0), fwker(1,1)
c
c     ---------------------------------------------------------------
c     Deconvolve
c     ---------------------------------------------------------------
c
ccc      rfac = 2.0d0/3.0d0/ms/mt
      rfac = 1.0d0/nj
      do k1 = 0, (ms-1)/2
      do k2 = 0, (mt-1)/2
         fk(k1,k2) = rfac*fw(k1,k2)/fwker(k1,k2)
         fk(-k1-1,k2) = rfac*fw(nf1-k1-1,k2)/fwker(nf1-k1-1,k2)
         fk(k1,-k2-1) = rfac*fw(k1,nf2-k2-1)/fwker(k1,nf2-k2-1)
         fk(-k1-1,-k2-1) = 
     1       rfac*fw(nf1-k1-1,nf2-k2-1)/fwker(nf1-k1-1,nf2-k2-1)
      enddo
      enddo
c
      return
      end
c
