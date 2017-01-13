cc Copyright (C) 2004-2009: Leslie Greengard and June-Yub Lee 
cc Contact: greengard@cims.nyu.edu
cc 
cc This software is being released under a FreeBSD license
cc (see license.txt in this directory). 
cc
c
c  NUFFT 1.3 release notes:
c
c  These codes are asymptotically fast (O(N log N)), but not optimized.
c
c  1) We initialize the FFT on every call.
c
c  2) We do not precompute the exponentials involved in "fast Gaussian
c  gridding".
c
c  3) We do not block structure the code so that irregularly placed points
c  are interpolated (gridded) in a cache-aware fashion.
c
c  4) We use the Netlib FFT library (www.netlib.org) 
c     rather than the state of the art FFTW package (www.fftw.org).
c
c  Different applications have different needs, and we have chosen
c  to provide the simplest code as a reasonable efficient template.
c
c**********************************************************************
      subroutine nufft1d1f90(nj,xj,cj,iflag,eps,ms,fk,ier)
      implicit none
      integer ier,iflag,istart,iw1,iwtot,iwsav
      integer j,jb1,jb1u,jb1d,k1,ms,next235,nf1,nj,nspread
      real*8 cross,cross1,diff1,eps,hx,pi,rat,r2lamb,t1,tau
      real*8 xc(-147:147),xj(nj)
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*16 cj(nj),fk(-ms/2:(ms-1)/2),zz,ccj
c ----------------------------------------------------------------------
      real*8, allocatable :: fw(:)
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
c     nj     number of sources   (integer)
c     xj     location of sources (real *8)
c
c            on interval [-pi,pi].
c
c     cj     strengths of sources (complex *16)
c     iflag  determines sign of FFT (see above)
c     eps    precision request  (between 1.0d-33 and 1.0d-1)
c               recomended value is 1d-15 for double precision calculations
c     ms     number of Fourier modes computed (-ms/2 to (ms-1)/2 )
c
c     OUTPUT:
c
c     fk     Fourier transform values (complex *16)
c     ier    error return code
c   
c            ier = 0  => normal execution.
c            ier = 1  => precision eps requested is out of range.
c
c     The type 1 NUFFT proceeds in three steps (see [GL]).
c
c     1) spread data to oversampled regular mesh using convolution with
c        a Gaussian 
c     2) compute FFT on uniform mesh
c     3) deconvolve each Fourier mode independently
c          (mutiplying by Fourier transform of Gaussian).
c
c ----------------------------------------------------------------------
c
c     The oversampled regular mesh is defined by 
c
c     nf1 = rat*ms  points, where rat is the oversampling ratio.
c       
c     For simplicity, we set  
c
c         rat = 2 for eps > 1.0d-11
c         rat = 3 for eps <= 1.0d-11.
c
c     The Gaussian used for convolution is:
c
c        g(x) = exp(-x^2 / 4tau) 
c
c     It can be shown [DR] that the precision eps is achieved when
c
c     nspread = int(-log(eps)/(pi*(rat-1d0)/(rat-.5d0)) + .5d0)
c     and tau is chosen as
c
c     tau = pi*lambda/(ms**2)
c     lambda = nspread/(rat(rat-0.5)).
c
c     Note that the Fourier transform of g(x) is
c
c     G(s) = exp(-s^2 tau) = exp(-pi*lambda s^2/ms^2)
c
c
c ----------------------------------------------------------------------
c     Fast Gaussian gridding is based on the following observation.
c
c     Let hx = 2*pi/nf1. In gridding data onto a regular mesh with
c     spacing nf1, we shift the source point xj by pi so 
c     that it lies in [0,2*pi] to simplify the calculations.
c     Since we are viewing the function
c     as periodic, this has no effect on the result.
c    
c     For source (xj+pi), let kb*hx denote the closest grid point and
c     let  kx*hx be a regular grid point within the spreading
c     distance. We can write
c
c     (xj+pi) - kx*hx = kb*hx + diff*hx - kx*hx = diff*hx - (kx-kb)*hx
c
c     where diff = (xj+pi)/hx - kb.
c
c     Let t1 = hx*hx/(4 tau) = pi/(nf1*nf1)/lambda*ms*ms
c                            = pi/lambda/(rat*rat)
c
c     exp(-( (xj+pi) -kx*hx)**2 / 4 tau)
c         = exp(-pi/lamb/rat^2 *(diff - (kx-kb))**2)
c         = exp(-t1 *(diff - (kx-kb))**2)
c         = exp(-t1*diff**2) * exp(2*t1*diff)**k * exp(-t1*k**2)
c           where k = kx-kb.
c 
c************************************************************************
c
c     Precision dependent parameters
c
c     rat is oversampling parameter
c     nspread is number of neighbors to which Gaussian gridding is
c     carried out.
c -------------------------------
      ier = 0
      if ((eps.lt.1d-33).or.(eps.gt.1d-1)) then
         ier = 1
         return
      endif
      if (eps.le.1d-11) then
         rat = 3.0d0
      else 
         rat = 2.0d0
      endif
      nspread = int(-log(eps)/(pi*(rat-1d0)/(rat-.5d0)) + .5d0)
      nf1 = rat*ms
      if (2*nspread.gt.nf1) then
         nf1 = next235(2d0*nspread) 
      endif 
c
c     lambda (described above) = nspread/(rat*(rat-0.5d0)) 
c     It is more convenient to define r2lamb = rat*rat*lambda
c
      r2lamb = rat*rat * nspread / (rat*(rat-.5d0))
      hx = 2*pi/nf1
c
c     -----------------------------------
c     Compute workspace size and allocate
c     -----------------------------------
      iw1 = 2*nf1
      iwsav = iw1+nspread+1
      iwtot = iwsav+4*nf1+15
      allocate ( fw(0:iwtot) )
c
c     ---------------------------------------------------------------
c     Precompute spreading constants and initialize fw
c     to hold one term needed for fast Gaussian gridding 
c     ---------------------------------------------------------------
      t1 = pi/r2lamb
      do k1 = 1, nspread
         fw(iw1+k1) = exp(-t1*k1**2)
      enddo
      call dcffti(nf1,fw(iwsav))
c
c     ---------------------------------------------------------------
c     Initialize fine grid data to zero.
c     ---------------------------------------------------------------
      do k1 = 0, 2*nf1-1
         fw(k1) = dcmplx(0d0,0d0)
      enddo
c
c     ---------------------------------------------------------------
c     Loop over sources (1,...,nj)
c
c     1. find closest mesh point (with periodic wrapping if necessary)
c     2. spread source data onto nearest nspread grid points
c        using fast Gaussian gridding.
c
c     The following is a little hard to read because it takes
c     advantage of fast gridding and optimized to minimize the 
c     the number of multiplies in the inner loops.
c
c    ---------------------------------------------------------------
c
      do j = 1, nj
         ccj = cj(j)/dble(nj)

         jb1 = int((xj(j)+pi)/hx)
         diff1 = (xj(j)+pi)/hx - jb1
         jb1 = mod(jb1, nf1)
         if (jb1.lt.0) jb1=jb1+nf1
c
         xc(0) = exp(-t1*diff1**2)
         cross = xc(0)
         cross1 = exp(2d0*t1 * diff1)
         do k1 = 1, nspread
            cross = cross * cross1
            xc(k1) = fw(iw1+k1)*cross
         enddo
         cross = xc(0)
         cross1 = 1d0/cross1
         do k1 = 1, nspread-1
            cross = cross * cross1
            xc(-k1) = fw(iw1+k1)*cross
         enddo
c
         jb1d = min(nspread-1, jb1)
         jb1u = min(nspread, nf1-jb1-1)
         do k1 = -nspread+1, -jb1d-1
	    istart = 2*(jb1+k1+nf1)
            zz=xc(k1)*ccj
            fw(istart)=fw(istart)+dreal(zz)
            fw(istart+1)=fw(istart+1)+dimag(zz)
         enddo
         do k1 = -jb1d, jb1u
	    istart = 2*(jb1+k1)
            zz=xc(k1)*ccj
            fw(istart)=fw(istart)+dreal(zz)
            fw(istart+1)=fw(istart+1)+dimag(zz)
         enddo
         do k1 = jb1u+1, nspread
	    istart = 2*(jb1+k1-nf1)
            zz=xc(k1)*ccj
            fw(istart)=fw(istart)+dreal(zz)
            fw(istart+1)=fw(istart+1)+dimag(zz)
         enddo
      enddo
c
c     ---------------------------------------------------------------
c     Compute 1D FFT and carry out deconvolution.
c 
c     There is a factor of (-1)**k1 needed to account for the 
c     FFT phase shift.
c     ---------------------------------------------------------------
c
      if (iflag .ge. 0) then
         call dcfftb(nf1,fw(0),fw(iwsav))
      else
         call dcfftf(nf1,fw(0),fw(iwsav))
      endif
c
      tau = pi * r2lamb / dble(nf1)**2
      cross1 = 1d0/sqrt(r2lamb)
      zz = dcmplx(fw(0),fw(1))
      fk(0) = cross1*zz
      do k1 = 1, (ms-1)/2
         cross1 = -cross1
         cross = cross1*exp(tau*dble(k1)**2)
	 zz = dcmplx(fw(2*k1),fw(2*k1+1))
         fk(k1) = cross*zz
	 zz = dcmplx(fw(2*(nf1-k1)),fw(2*(nf1-k1)+1))
         fk(-k1) = cross*zz
      enddo
      if (ms/2*2.eq.ms) then
         cross = -cross1*exp(tau*dble(ms/2)**2)
         zz = dcmplx(fw(2*nf1-ms),fw(2*nf1-ms+1))
         fk(-ms/2) = cross*zz
      endif
      deallocate(fw)
      return
      end
c
c
c
c
c
************************************************************************
      subroutine nufft1d2f90(nj,xj,cj, iflag,eps, ms,fk,ier)
      implicit none
      integer ier,iflag,iw1,iwsav,iwtot,j,jb1,jb1u,jb1d,k1
      integer ms,next235,nf1,nj,nspread,nw
      real*8 cross,cross1,diff1,eps,hx,pi,rat,r2lamb,t1
      real*8 xj(nj),xc(-147:147)
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*16 cj(nj), fk(-ms/2:(ms-1)/2)
      complex*16 zz
c ----------------------------------------------------------------------
      real*8, allocatable :: fw(:)
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
c
c     The type 2 algorithm proceeds in three steps (see [GL]).
c
c     1) deconvolve (amplify) each Fourier mode first 
c     2) compute inverse FFT on uniform fine grid
c     3) spread data to regular mesh using Gaussian
c
c
c     See subroutine nufft1d1f90(nj,xj,cj,iflag,eps,ms,fk,ier)
c     for more comments on fast gridding and parameter selection.
c
************************************************************************
c
c     Precision dependent parameters
c
c     rat is oversampling parameter
c     nspread is number of neighbors to which Gaussian gridding is
c     carried out.
c     -------------------------------
c
c     lambda (described above) = nspread/(rat*(rat-0.5d0)) 
c     It is more convenient to define r2lamb = rat*rat*lambda
c
c     -------------------------------
      ier = 0
      if ((eps.lt.1d-33).or.(eps.gt.1d-1)) then
         ier = 1
         return
      endif
      if (eps.le.1d-11) then
         rat = 3.0d0
      else 
         rat = 2.0d0
      endif
c
      nspread = int(-log(eps)/(pi*(rat-1d0)/(rat-.5d0)) + .5d0)
      nf1 = rat*ms
      if (2*nspread.gt.nf1) then
         nf1 = next235(2d0*nspread) 
      endif 
c
      r2lamb = rat*rat * nspread / (rat*(rat-.5d0))
      hx = 2*pi/nf1
c
c     -----------------------------------
c     Compute workspace size and allocate
c     -----------------------------------
      iw1 = 2*nf1
      iwsav = iw1 + nspread+1
      iwtot = iwsav + 4*nf1 + 15
      allocate ( fw(0:iwtot))
c
c     ---------------------------------------------------------------
c     Precompute spreading constants and initialize fw
c     to hold one term needed for fast Gaussian gridding 
c     ---------------------------------------------------------------
      t1 = pi/r2lamb
      do k1 = 1, nspread
         fw(iw1+k1) = exp(-t1*k1**2)
      enddo
      call dcffti(nf1,fw(iwsav))
c
c     ---------------------------------------------------------------
c     Deconvolve and compute inverse 1D FFT
c     (A factor of (-1)**k is needed to shift phase.)
c     ---------------------------------------------------------------
c
      t1 = pi * r2lamb / dble(nf1)**2
      cross1 = 1d0/sqrt(r2lamb)
      zz = cross1*fk(0)
      fw(0) = dreal(zz)
      fw(1) = dimag(zz)
      do k1 = 1, (ms-1)/2
         cross1 = -cross1
         cross = cross1*exp(t1*dble(k1)**2)
         zz = cross*fk(k1)
         fw(2*k1) = dreal(zz)
         fw(2*k1+1) = dimag(zz)
         zz = cross*fk(-k1)
         fw(2*(nf1-k1)) = dreal(zz)
         fw(2*(nf1-k1)+1) = dimag(zz)
      enddo
      cross = -cross1*exp(t1*dble(ms/2)**2)
      if (ms/2*2.eq.ms) then
	 zz = cross*fk(-ms/2)
         fw(2*nf1-ms) = dreal(zz)
         fw(2*nf1-ms+1) = dimag(zz)
      endif
      do k1 = (ms+1)/2, nf1-ms/2-1
         fw(2*k1) = dcmplx(0d0, 0d0)
         fw(2*k1+1) = dcmplx(0d0, 0d0)
      enddo
c
      if (iflag .ge. 0) then
         call dcfftb(nf1,fw(0),fw(iwsav))
      else
         call dcfftf(nf1,fw(0),fw(iwsav))
      endif
c
c     ---------------------------------------------------------------
c     Loop over target points (1,...,nj)
c
c       1. find closest mesh point (with periodic wrapping if needed)
c       2. get contributions from regular fine grid to target
c          locations using Gaussian convolution.
c     ---------------------------------------------------------------
      t1 = pi/r2lamb
      do j = 1, nj
         cj(j) = dcmplx(0d0,0d0)
         jb1 = int((xj(j)+pi)/hx)
         diff1 = (xj(j)+pi)/hx - jb1
         jb1 = mod(jb1, nf1)
         if (jb1.lt.0) jb1=jb1+nf1
         xc(0) = exp(-t1*diff1**2)
         cross = xc(0)
         cross1 = exp(2d0*t1 * diff1)
         do k1 = 1, nspread
            cross = cross * cross1
            xc(k1) = fw(iw1+k1)*cross
         enddo
         cross = xc(0)
         cross1 = 1d0/cross1
         do k1 = 1, nspread-1
            cross = cross * cross1
            xc(-k1) = fw(iw1+k1)*cross
         enddo
c
         jb1d = min(nspread-1, jb1)
         jb1u = min(nspread, nf1-jb1-1)
         do k1 = -nspread+1, -jb1d-1
	    zz = dcmplx(fw(2*(jb1+k1+nf1)),fw(2*(jb1+k1+nf1)+1))
            cj(j) = cj(j) + xc(k1)*zz
         enddo
         do k1 = -jb1d, jb1u
	    zz = dcmplx(fw(2*(jb1+k1)),fw(2*(jb1+k1)+1))
            cj(j) = cj(j) + xc(k1)*zz
         enddo
         do k1 = jb1u+1, nspread
	    zz = dcmplx(fw(2*(jb1+k1-nf1)),fw(2*(jb1+k1-nf1)+1))
            cj(j) = cj(j) + xc(k1)*zz
         enddo
      enddo
      deallocate(fw)
      return
      end
c
c
c
c
c
c
************************************************************************
      subroutine nufft1d3f90(nj,xj,cj, iflag,eps, nk,sk,fk,ier)
      implicit none
      integer ier,iw1,iwsave,iwtot,j,jb1,k1,kb1,kmax,nj,iflag,nk
      integer next235,nf1,nspread
      real*8 ang,cross,cross1,diff1,eps,hx,hs,rat,pi,r2lamb1
      real*8 sm,sb,t1,t2,xm,xb
      real*8 xc(-147:147), xj(nj), sk(nk)
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*16 cj(nj), fk(nk), zz, cs
c
c ----------------------------------------------------------------------
      integer nw, istart
      real*8, allocatable :: fw(:)
c ----------------------------------------------------------------------
c     if (iflag .ge. 0) then
c
c                  nj
c     fk(sk(k)) = SUM cj(j) exp(+i sk(k) xj(j))  for k = 1,..., nk
c                 j=1                            
c
c     else
c
c                  nj
c     fk(sk(k)) = SUM cj(j) exp(-i sk(k) xj(j))  for k = 1,..., nk
c                 j=1                            
c ----------------------------------------------------------------------
c     INPUT:
c
c     nj     number of sources   (integer)
c     xj     location of sources (double array)
c     cj     strengths of sources (double complex array)
c     iflag  determines sign of FFT (see above)
c     eps    precision request  (between 1.0d-33 and 1.0d-1)
c               recomended value is 1d-15 for double precision calculations
c     nk     number of (noninteger) Fourier modes computed
c     sk     k-values (locations) of desired Fourier modes
c
c     OUTPUT:
c
c     fk     Fourier transform values (double complex array)
c     ier    error return code
c   
c            ier = 0  => normal execution.
c            ier = 1  => precision eps requested is out of range.
c
c
c     References:
c
c     [DR] Fast Fourier transforms for nonequispaced data,
c          A. Dutt and V. Rokhlin, SIAM J. Sci. Comput. 14, 
c          1368-1383, 1993.
c
c     [LG] The type 3 nonuniform FFT and its applications
c          J.-Y. Lee and L. Greengard, J. Comput. Phys. 206, 1-5 (2005).
c
c     The algorithm is essentially a concatenation of the 
c     type 1 and 2 transforms.
c
c     1) Gaussian gridding of strengths cj(j) centered at xj 
c        to create f_\tau(n \Delta_x)  (in notation of [LG])
c     2) Deconvolve each regular grid mode 
c        to create f^{-\sigma}_\tau(n \Delta_x)  (in notation of [LG])
c     3) compute FFT on uniform mesh
c        to create F^{-\sigma}_\tau(m \Delta_s)  (in notation of [LG])
c     4) Gaussian gridding to irregular frequency points
c        to create F_\tau(s_k)  (in notation of [LG])
c     5) Deconvolution of  result 
c        to create F(s_k)  (in notation of [LG])
c
c***********************************************************************
      ier = 0
      if ((eps.lt.1d-33).or.(eps.gt.1d-1)) then
         ier = 1
         return
      endif
c
c --- Check the ranges of {xj}/{sk} and the workspace size
c
      t1 = xj(1)
      t2 = xj(1)
      do j = 2, nj
         if (xj(j).gt.t2) then
             t2=xj(j)
         else if (xj(j).lt.t1) then
             t1=xj(j)
         endif
      enddo
      xb = (t1+t2) / 2d0
      xm = max(t2-xb,-t1+xb)  ! max(abs(t2-xb),abs(t1-xb))
c
      t1 = sk(1)
      t2 = sk(1)
      do k1 = 2, nk
         if (sk(k1).gt.t2) then
             t2=sk(k1)
         else if (sk(k1).lt.t1) then
             t1=sk(k1)
         endif
      enddo
      sb = (t1+t2) / 2d0
      sm = max(t2-sb,-t1+sb)
c
c     -------------------------------
c     Precision dependent parameters
c
c     rat is oversampling parameter
c     nspread is number of neighbors to which Gaussian gridding is
c     carried out.
c     -------------------------------
      if (eps.le.1d-11) then
         rat = sqrt(3.0d0)
      else 
         rat = sqrt(2.0d0)
      endif
c
      nspread = int(-log(eps)/(pi*(rat-1d0)/(rat-.5d0)) + .5d0)
      t1 = 2d0/pi * xm*sm
      nf1 = next235(rat*max(rat*t1+2*nspread,2*nspread/(rat-1)))
      rat = (sqrt(nf1*t1+nspread**2)-nspread)/t1
c
      r2lamb1 = rat*rat * nspread / (rat*(rat-.5d0))
      hx = pi/(rat*sm)
      hs = 2d0*pi/dble(nf1)/hx            ! hx hs = 2.pi/nf1
c
c     -------------------------------
c     Compute workspace size and allocate
c     -------------------------------
c
      kmax = int(nf1*(r2lamb1-nspread)/r2lamb1+.1d0)
      iw1 = 2*nf1
      iwsave = iw1 + nspread+1
      iwtot = iwsave + 16+4*nf1
      allocate ( fw(0:iwtot-1) )
c
c     ---------------------------------------------------------------
c     Precompute spreading constants and initialize fw
c     to hold one term needed for fast Gaussian gridding 
c     ---------------------------------------------------------------
c
      t1 = pi/r2lamb1
      do k1 = 1, nspread
         fw(iw1+k1) = exp(-t1*k1**2)
      enddo
c
      call dcffti(nf1,fw(iwsave))
c
c     ---------------------------------------------------------------
c     Initialize fine grid data to zero.
c     ---------------------------------------------------------------
      do k1 = 0, 2*nf1-1
         fw(k1) = dcmplx(0d0,0d0)
      enddo
c
c     ---------------------------------------------------------------
c     Step 1/5  - gridding as in type 1 transform.
c     ---------------------------------------------------------------
c
      t1 = pi/r2lamb1
      if (iflag .lt. 0) sb = -sb
      do j = 1, nj
         jb1 = int(dble(nf1/2) + (xj(j)-xb)/hx)
         diff1 = dble(nf1/2) + (xj(j)-xb)/hx - jb1
         ang = sb*xj(j)
         cs = dcmplx(cos(ang),sin(ang)) * cj(j)

         xc(0) = exp(-t1*diff1**2)
         cross = xc(0)
         cross1 = exp(2d0*t1 * diff1)
         do k1 = 1, nspread
            cross = cross * cross1
            xc(k1) = fw(iw1+k1)*cross
         enddo
         cross = xc(0)
         cross1 = 1d0/cross1
         do k1 = 1, nspread-1
            cross = cross * cross1
            xc(-k1) = fw(iw1+k1)*cross
         enddo

         do k1 = -nspread+1, nspread
	    istart = 2*(jb1+k1)
	    zz = xc(k1)*cs
            fw(istart) = fw(istart) + dreal(zz)
            fw(istart+1) = fw(istart+1) + dimag(zz)
         enddo
      enddo
      if (iflag .lt. 0) sb = -sb
c
c ---------------------------------------------------------------
c     Step 2: Deconvolve (amplify) as in Type 2 transform 
c     Step 3: Compute FFT with shift
c             (-1)^k F_(k+M/2) = Sum (-1)^j F_(j+M/2) e(2pi ijk/M)
c ---------------------------------------------------------------
c
      t1 = pi * r2lamb1 / dble(nf1)**2
      cross1 = (1d0-2d0*mod(nf1/2,2))/r2lamb1
      zz = dcmplx(fw(nf1),fw(nf1+1))
      zz = cross1*zz
      fw(nf1) = dreal(zz)
      fw(nf1+1) = dimag(zz)
      do k1 = 1, kmax
         cross1 = -cross1
         cross = cross1*exp(t1*dble(k1)**2)
         zz = dcmplx(fw(nf1-2*k1),fw(nf1-2*k1+1))
         zz = cross*zz
         fw(nf1-2*k1) = dreal(zz)
         fw(nf1-2*k1+1) = dimag(zz)
         zz = dcmplx(fw(nf1+2*k1),fw(nf1+2*k1+1))
         zz = cross*zz
         fw(nf1+2*k1) = dreal(zz)
         fw(nf1+2*k1+1) = dimag(zz)
      enddo
c
      if (iflag .ge. 0) then
         call dcfftb(nf1,fw(0),fw(iwsave))
      else
         call dcfftf(nf1,fw(0),fw(iwsave))
      endif
      do k1 = 1, kmax+nspread, 2
         fw(nf1+2*k1) = -fw(nf1+2*k1)
         fw(nf1+2*k1+1) = -fw(nf1+2*k1+1)
         fw(nf1-2*k1) = -fw(nf1-2*k1)
         fw(nf1-2*k1+1) = -fw(nf1-2*k1+1)
      enddo
c
c     ---------------------------------------------------------------
c     Step 4 Gaussian gridding to irregular points
c     Step 5 Final deconvolution
c     ---------------------------------------------------------------
      t1 = pi/r2lamb1
      do j = 1, nk
         kb1 = int(dble(nf1/2) + (sk(j)-sb)/hs)
         diff1 = dble(nf1/2) + (sk(j)-sb)/hs - kb1

         ! exp(-t1*(diff1-k1)**2) = xc(k1)
         xc(0) = exp(-t1*diff1**2)
         cross = xc(0)
         cross1 = exp(2d0*t1 * diff1)
         do k1 = 1, nspread
            cross = cross * cross1
            xc(k1) = fw(iw1+k1)*cross
         enddo
         cross = xc(0)
         cross1 = 1d0/cross1
         do k1 = 1, nspread-1
            cross = cross * cross1
            xc(-k1) = fw(iw1+k1)*cross
         enddo
c
         fk(j) = dcmplx(0d0,0d0)
         do k1 = -nspread+1, nspread
	    zz = dcmplx(fw(2*(kb1+k1)),fw(2*(kb1+k1)+1))
            fk(j) = fk(j) + xc(k1)*zz
         enddo
      enddo
c
      if (iflag .lt. 0) xb = -xb
      t1 = r2lamb1/(4d0*pi) * hx**2
      do j = 1, nk
         fk(j) = (exp(t1*(sk(j)-sb)**2))*fk(j)
         ang = (sk(j)-sb)*xb
         fk(j) = dcmplx(cos(ang),sin(ang)) * fk(j)
      enddo
      deallocate(fw)
      return
      end
