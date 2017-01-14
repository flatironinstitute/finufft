c**********************************************************************
      subroutine nufft3d1f90x(nj,xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk,ier)
      implicit none
      integer ier,iflag,istart,iw1,iwtot,iwsav,n1,itype,mu,nspread
      integer j,jb1,jb1u,jb1d,k1,k2,k3,ms,mt,next235,nf1,nf2,nf3,nj
      real*8 cross,cross1,diff1,eps,hx,pi,rat,r2lamb,t1,tau
      real*8 xj(nj),yj(nj),zj(nj),xker,yker,zker,rfac
      real*8 params(4)
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*16 cj(nj),fk(-ms/2:(ms-1)/2,-mt/2:(mt-1)/2,-mu/2:(mu-1)/2)
      complex *16 zz,ccj,cker
c ----------------------------------------------------------------------
      complex*16, allocatable :: fw(:,:,:)
      complex*16, allocatable :: fwker(:,:,:)
      complex*16, allocatable :: fwsav1(:)
      complex*16, allocatable :: fwsav2(:)
      complex*16, allocatable :: fwsav3(:)
c ----------------------------------------------------------------------
c
c    xxxxx
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
      nf3 = 2*mu
ccc      if (2*params(2).gt.nf1) then
ccc         nf1 = next235(2d0*params(2)) 
ccc      endif 
ccc      if (2*params(2).gt.nf2) then
ccc         nf2 = next235(2d0*params(2)) 
ccc      endif 
      call prinf(' nf1 is *',nf1,1)
      call prinf(' nf2 is *',nf2,1)
c
      allocate(fw(0:nf1-1,0:nf2-1,0:nf3-1))
      allocate(fwker(0:nf1-1,0:nf2-1,0:nf3-1))
      allocate(fwsav1(4*nf1+15))
      allocate(fwsav2(4*nf2+15))
      allocate(fwsav3(4*nf3+15))
      call dcffti(nf1,fwsav1)
      call dcffti(nf2,fwsav2)
      call dcffti(nf3,fwsav3)
c
      itype = 1
      call tempspread3d(nf1,nf2,nf3,fw,nj,xj,yj,zj,cj,itype,params)
      n1 = 1
      xker = 0.0d0
      yker = 0.0d0
      zker = 0.0d0
      cker = 1.0d0
      call tempspread3d(nf1,nf2,nf3,fwker,n1,xker,yker,zker,cker,
     1                  itype,params)
c
c     ---------------------------------------------------------------
c     Call 2D FFT 
c     ---------------------------------------------------------------
c
      call dcfft3d(iflag,nf1,nf2,nf3,fw,fwsav1,fwsav2,fwsav3)
      call dcfft3d(iflag,nf1,nf2,nf3,fwker,fwsav1,fwsav2,fwsav3)
c
c     ---------------------------------------------------------------
c     Deconvolve
c     ---------------------------------------------------------------
c
      rfac = 1.0d0/nj
      do k1 = 0, (ms-1)/2
      do k2 = 0, (mt-1)/2
      do k3 = 0, (mu-1)/2
         fk(k1,k2,k3) = rfac*fw(k1,k2,k3)/fwker(k1,k2,k3)
         fk(-k1-1,k2,k3)=rfac*fw(nf1-k1-1,k2,k3)/fwker(nf1-k1-1,k2,k3)
         fk(k1,-k2-1,k3)=rfac*fw(k1,nf2-k2-1,k3)/fwker(k1,nf2-k2-1,k3)
         fk(-k1-1,-k2-1,k3) = 
     1       rfac*fw(nf1-k1-1,nf2-k2-1,k3)/fwker(nf1-k1-1,nf2-k2-1,k3)
c
         fk(k1,k2,-k3-1)=rfac*fw(k1,k2,nf3-k3-1)/fwker(k1,k2,nf3-k3-1)
         fk(-k1-1,k2,-k3-1)=
     1       rfac*fw(nf1-k1-1,k2,nf3-k3-1)/fwker(nf1-k1-1,k2,nf3-k3-1)
         fk(k1,-k2-1,-k3-1)=
     1       rfac*fw(k1,nf2-k2-1,nf3-k3-1)/fwker(k1,nf2-k2-1,nf3-k3-1)
         fk(-k1-1,-k2-1,-k3-1) = rfac*fw(nf1-k1-1,nf2-k2-1,nf3-k3-1)
     1       /fwker(nf1-k1-1,nf2-k2-1,nf3-k3-1)
      enddo
      enddo
      enddo
c
      return
      end
c
