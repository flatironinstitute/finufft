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
      subroutine nufft3d1f90(nj,xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk,ier)
      implicit none
      integer ier,i3,iflag,ii,istart,is2,i2(-147:147)
      integer iw10,iw11,iw12,iw13,iw14,iw15,iw16,iwtot
      integer j,jb1,jb1u,jb1d,jb2,jb3,k1,k2,k3,ms,mt,mu,next235
      integer nf1,nf2,nf3,nj,nspread,nw1,nw2,nw3
      real*8  xj(nj),yj(nj),zj(nj)
      real*8  eps,hx,hy,hz,pi,rat,r2lamb,t1,t2,t3,diff1,diff2,diff3
      real*8  cross,cross1,xc(-147:147),yc(-147:147),zc(-147:147)
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*16 cj(nj),fk(-ms/2:(ms-1)/2,-mt/2:(mt-1)/2,-mu/2:(mu-1)/2)
      complex*16 cc,c2,ccj,zz
c ----------------------------------------------------------------------
      real*8, allocatable :: fw(:)
c ----------------------------------------------------------------------
c
c     if (iflag .ge. 0) then
c
c                  1  nj
c     fk(k1,k2) = -- SUM cj(j) exp(+i (k1,k2,k3)*(xj(j),yj(j),zj(j)))   
c                 nj j=1 
c                                          for -ms/2 <= k1 <= (ms-1)/2
c                                          for -mt/2 <= k1 <= (mt-1)/2
c                                          for -mu/2 <= k1 <= (mu-1)/2
c     else
c
c                  1  nj
c     fk(k1,k2) = -- SUM cj(j) exp(-i (k1,k2,k3)*(xj(j),yj(j),zj(j)))   
c                 nj j=1 
c                                          for -ms/2 <= k1 <= (ms-1)/2
c                                          for -mt/2 <= k1 <= (mt-1)/2
c                                          for -mu/2 <= k1 <= (mu-1)/2
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
c     nj       number of sources   (integer)
c     xj,yj,zj location of sources (real *8 arrays)
c     cj       strengths of sources (complex *16 array)
c     iflag    determines sign of FFT (see above)
c     eps      precision request  (between 1.0d-33 and 1.0d-1)
c               recomended value is 1d-15 for double precision calculations
c     ms       number of Fourier modes computed (first index)
c     mt       number of Fourier modes computed (second index)
c     mu       number of Fourier modes computed (third index)
c
c     OUTPUT:
c
c     fk       Fourier transform values (2D complex *16 array)
c     ier      error return code
c
c              ier = 0 => normal execution
c              ier = 1 => precision requested is out of range.
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
c     nf2 = rat*mt  points, where rat is the oversampling ratio.
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
c     In 2D/3D, this decomposition is carried out in each dimension.
c ----------------------------------------------------------------------
c     Precision dependent parameters
c
c     rat is oversampling arameter
c     nspread is number of neighbors to which Gaussian gridding is
c     carried out in each dimension.
c -------------------------------
      ier = 0
      if (eps.lt.1d-33 .or. eps.gt.1d-1) then
	 ier = 1
	 return
      endif
      if (eps.le.1d-11) then
         rat = 3.0d0
      else
         rat = 2.0d0
      endif
      nspread = int(-log(eps)/(pi*(rat-1d0)/(rat-.5d0)) + .5d0) + 1
      nf1 = rat*ms
      nf2 = rat*mt
      nf3 = rat*mu
      if (2*nspread.gt.min(nf1,nf2,nf3)) then
         ii = next235(2d0*nspread)
         nf1 = max(nf1, ii)
         nf2 = max(nf2, ii)
         nf3 = max(nf3, ii)
      endif
c
c     lambda (described above) = nspread/(rat*(rat-0.5d0))
c     It is more convenient to define r2lamb = rat*rat*lambda
c
      r2lamb = rat*rat * nspread / (rat*(rat-.5d0))
      hx = 2*pi/nf1
      hy = 2*pi/nf2
      hz = 2*pi/nf3
c
c     -------------------------------
c     Compute workspace size and allocate
c
c     nw1 for fine grid, nw2 for second/thrid index, nw3 for fft.
c     -------------------------------
      nw1 = nf1*nf2*nf3 
      nw2 = nw1 + max(nf2,nf3)
      nw3 = nw2 + max(nf1,nf2,nf3)
c
      iw10 = 2*nw3
      iw11 = iw10 + ms
      iw12 = iw11 + mt
      iw13 = iw12 + mu/2 + 1
      iw14 = iw13 + 48+16+2*nf1
      iw15 = iw14 + 48+16+2*nf2
      iw16 = iw15 + 48+16+2*nf3
      allocate( fw(0:iw16-1) )
c
c     ---------------------------------------------------------------
c     Precompute spreading constants and Initialize rwork
c     ---------------------------------------------------------------
      t1 = pi/r2lamb
      t2 = pi/r2lamb
      t3 = pi/r2lamb
      do k1 = 1, nspread
         fw(iw13+k1) = exp(-t1*k1**2)
         fw(iw14+k1) = exp(-t2*k1**2)
         fw(iw15+k1) = exp(-t3*k1**2)
      enddo

      call dcfti1(nf1,fw(iw13+64),fw(iw13+48))
      call dcfti1(nf2,fw(iw14+64),fw(iw14+48))
      call dcfti1(nf3,fw(iw15+64),fw(iw15+48))
c
c     ---------------------------------------------------------------
c     Precompute deconvolution contants
c     ---------------------------------------------------------------
      t1 = pi * r2lamb / dble(nf1)**2
      cross1 = 1d0-2d0*mod(ms/2,2)
      do k1 = -ms/2, (ms-1)/2
         fw(iw10+k1+ms/2)=cross1*exp(t1*dble(k1)**2)
         cross1 = -cross1
      enddo
      t2 = pi * r2lamb / dble(nf2)**2
      cross1 = 1d0-2d0*mod(mt/2,2)
      do k1 = -mt/2, (mt-1)/2
         fw(iw11+k1+mt/2)=cross1*exp(t2*dble(k1)**2)
         cross1 = -cross1
      enddo
      t3 = pi * r2lamb / dble(nf3)**2
      cross1 = 1d0 / (sqrt(r2lamb)*r2lamb)
      do k1 = 0, mu/2
         fw(iw12+k1)=cross1*exp(t3*dble(k1)**2)
         cross1 = -cross1
      enddo
c
c     ---------------------------------------------------------------
c     Initialize fine grid to zero.
c     ---------------------------------------------------------------
      do k1 = 0, 2*nf1*nf2*nf3-1
         fw(k1) = 0d0
      enddo
c
c     ---------------------------------------------------------------
c     Loop over sources (1,...,nj)
c
c       1. find closest mesh point (with periodic wrapping if necessary)
c       2. spread source data onto nearest nspread**3 grid points
c          using fast Gaussian gridding.
c
c     Code is not easy to read because fast gridding is optimized to 
c     minimize the number of multiplies in inner loop and there is a
c     certain amout of unrolling involved.
c     ---------------------------------------------------------------
      t1 = pi/r2lamb
      t2 = pi/r2lamb
      t3 = pi/r2lamb
      do j = 1, nj
         ccj = cj(j)/dble(nj)
c
         jb1 = int((xj(j)+pi)/hx)
         diff1 = (xj(j)+pi)/hx - jb1
         jb1 = mod(jb1, nf1)
         if (jb1.lt.0) jb1=jb1+nf1
c
         jb2 = int((yj(j)+pi)/hy)
         diff2 = (yj(j)+pi)/hy - jb2
         jb2 = mod(jb2, nf2)
         if (jb2.lt.0) jb2=jb2+nf2
c
         jb3 = int((zj(j)+pi)/hz)
         diff3 = (zj(j)+pi)/hz - jb3
         jb3 = mod(jb3, nf3)
         if (jb3.lt.0) jb3=jb3+nf3
c
c     exp(-t1*(diff1-k1)**2) = xc(k1) / exp(-t1*(diff2**2+diff3**2))
c
         xc(0) = exp(-t1*diff1**2-t2*diff2**2-t3*diff3**2)
         cross = xc(0)
         cross1 = exp(2d0*t1 * diff1)
         do k1 = 1, nspread
            cross = cross * cross1
            xc(k1) = fw(iw13+k1)*cross
         enddo
         cross = xc(0)
         cross1 = 1d0/cross1
         do k1 = 1, nspread-1
            cross = cross * cross1
            xc(-k1) = fw(iw13+k1)*cross
         enddo
c
c     exp(-t1*(diff2-k2)**2) = yc(k2) * exp(-t1*diff2**2)
c
         yc(0) = 1d0
         cross = exp(2d0*t2 * diff2)
         cross1 = cross
         do k2 = 1, nspread-1
            yc(k2) = fw(iw14+k2)*cross
            yc(-k2) = fw(iw14+k2)/cross
            cross = cross * cross1
         enddo
         yc(nspread) = fw(iw14+nspread)*cross
c
c     exp(-t1*(diff3-k3)**2) = zc(k3) * exp(-t1*diff3**2)
c
         zc(0) = 1d0
         cross = exp(2d0*t3 * diff3)
         cross1 = cross
         do k3 = 1, nspread-1
            zc(k3) = fw(iw15+k3)*cross
            zc(-k3) = fw(iw15+k3)/cross
            cross = cross * cross1
         enddo
         zc(nspread) = fw(iw15+nspread)*cross
c
         jb1d = min(nspread-1, jb1)
         jb1u = min(nspread, nf1-jb1-1)
         do k2 = -nspread+1, nspread
            i2(k2) = jb2+k2
            if (i2(k2).lt.0) then
               i2(k2) = i2(k2) + nf2
            elseif (i2(k2).ge.nf2) then
               i2(k2) = i2(k2) - nf2
            endif
         enddo
c
         do k3 = -nspread+1, nspread
            i3 = jb3+k3
            if (i3.lt.0) then
               i3 = i3 + nf3
            elseif (i3.ge.nf3) then
               i3 = i3 - nf3
            endif
c
            c2 = zc(k3)*ccj
            do k2 = -nspread+1, nspread
               cc = yc(k2)*c2
               ii = jb1 + i2(k2)*nf1 + i3*nf1*nf2 ! cfine(ib,jb+k2,kb+k3)
               do k1 = -nspread+1, -jb1d-1
		  istart = 2*(k1+nf1+ii)
		  zz = xc(k1)*cc
                  fw(istart) = fw(istart) + dreal(zz)
                  fw(istart+1) = fw(istart+1) + dimag(zz)
               enddo
               do k1 = -jb1d, jb1u
		  istart = 2*(k1+ii)
		  zz = xc(k1)*cc
                  fw(istart) = fw(istart) + dreal(zz)
                  fw(istart+1) = fw(istart+1) + dimag(zz)
               enddo
               do k1 = jb1u+1, nspread
		  istart = 2*(k1-nf1+ii)
		  zz = xc(k1)*cc
                  fw(istart) = fw(istart) + dreal(zz)
                  fw(istart+1) = fw(istart+1) + dimag(zz)
               enddo
            enddo
         enddo
      enddo
c
c     ---------------------------------------------------------------
c     Compute 3D FFT and carry out deconvolution.
c
c     There are factors of form (-1)**k1 to account for FFT phase
c     shift.
c     ---------------------------------------------------------------
      i3 = iw13 + 48
      do k3 = 0, nf3-1
         do k2 = 0, nf2-1
            ii = nf1 * (k2+k3*nf2)  ! cfine(0,k2,k3) = cw(nf1*(k2+k3*nf2))
            if (iflag .ge. 0) then
               call dcftb1(nf1,fw(2*ii),fw(2*nw2),fw(i3+16),fw(i3))
            else
               call dcftf1(nf1,fw(2*ii),fw(2*nw2),fw(i3+16),fw(i3))
            endif
         enddo
      enddo
c
      i3 = iw14 + 48
      do k3 = 0, nf3-1
         do k1 = 0, nf1-1
            ii = k1 + k3 * nf1*nf2  ! cfine(k1,0,k3) = cw(k1+k3*nf1*nf2)
            do k2 = 0, nf2-1
               istart = 2*(nw1+k2)
               is2 = 2*(ii + k2*nf1)
               fw(istart) = fw(is2)
               fw(istart+1) = fw(is2+1)
            enddo
            if (iflag .ge. 0) then
               call dcftb1(nf2,fw(2*nw1),fw(2*nw2),fw(i3+16),fw(i3))
            else
               call dcftf1(nf2,fw(2*nw1),fw(2*nw2),fw(i3+16),fw(i3))
            endif
            do k2 = 0, nf2-1
               istart = 2*(ii + k2*nf1)
               is2 = 2*(nw1+k2)
               fw(istart) = fw(is2)
               fw(istart+1) = fw(is2+1)
            enddo
         enddo
      enddo
c
      i3 = iw15+48
      do k2 = -mt/2,  (mt-1)/2
         do k1 = -ms/2, (ms-1)/2
            ii = k1
            if (k1.lt.0) ii = ii+nf1
            ii = ii + k2*nf1
            if (k2.lt.0) ii = ii+nf1*nf2
c
            do k3 = 0, nf3-1
               istart = 2*(nw1+k3)
               is2 = 2*(ii + k3*nf1*nf2)
               fw(istart) = fw(is2)
               fw(istart+1) = fw(is2+1)
            enddo
            if (iflag .ge. 0) then
               call dcftb1(nf3,fw(2*nw1),fw(2*nw2),fw(i3+16),fw(i3))
            else
               call dcftf1(nf3,fw(2*nw1),fw(2*nw2),fw(i3+16),fw(i3))
            endif
c
            cross = fw(iw10+k1+ms/2) * fw(iw11+k2+mt/2)
	    zz = dcmplx(fw(2*nw1),fw(2*nw1+1))
            fk(k1, k2, 0) = (cross*fw(iw12))*zz
            do k3 = 1, (mu-1)/2
               istart = 2*(nw1+k3)
	       zz = dcmplx(fw(istart),fw(istart+1))
               fk(k1,k2,k3) = (cross*fw(iw12+k3))*zz
               istart = 2*(nw1+nf3-k3)
	       zz = dcmplx(fw(istart),fw(istart+1))
               fk(k1,k2,-k3) = (cross*fw(iw12+k3))*zz
            enddo
            if (mu/2*2.eq.mu) then 
               istart = 2*(nw1+nf3-mu/2)
	       zz = dcmplx(fw(istart),fw(istart+1))
     	       fk(k1,k2,-mu/2) =
     &           (cross*fw(iw12+mu/2))*zz
	    endif
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
      subroutine nufft3d2f90(nj,xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk,ier)
      implicit none
      integer ii,i3,ier,iflag,istart,is2
      integer iw10,iw11,iw12,iw13,iw14,iw15,iw16,j,jb1,jb2,jb3,k1,k2,k3
      integer ms,mt,mu,next235,nf1,nf2,nf3,nj,nspread,nw1,nw2,nw3
      integer i2(-147:147),jb1u, jb1d
      real*8  xj(nj), yj(nj), zj(nj)
      real*8 eps,pi,rat,t1,t2,t3,diff1,diff2,diff3,hx,hy,hz,r2lamb
      real*8 cross, cross1,xc(-147:147),yc(-147:147),zc(-147:147)
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*16 cj(nj),fk(-ms/2:(ms-1)/2,-mt/2:(mt-1)/2,-mu/2:(mu-1)/2)
      complex*16 cc, c2, zz
c ----------------------------------------------------------------------
      real*8, allocatable :: fw(:)
c ----------------------------------------------------------------------
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
c ----------------------------------------------------------------------
c     INPUT:
c
c     nj     number of output values   (integer)
c     xj,yj,zj location of output values (real *8 arrays)
c     iflag  determines sign of FFT (see above)
c     eps    precision request  (between 1.0d-33 and 1.0d-1)
c               recomended value is 1d-15 for double precision calculations
c     ms     number of Fourier modes (1st index)  [-ms/2:(ms-1)/2]
c     mt     number of Fourier modes (2nd index)  [-mt/2:(mt-1)/2]
c     mu     number of Fourier modes (3nd index)  [-mu/2:(mu-1)/2]
c     fk     Fourier coefficient values (2D complex *16 array)
c
c     OUTPUT:
c
c     cj     output values (complex *16 array)
c     ier    error return code
c   
c            ier = 0  => normal execution.
c            ier = 1  => precision eps requested is out of range.
c
c ----------------------------------------------------------------------
c
c     The type 2 algorithm proceeds in three steps (see [GL]).
c
c     1) deconvolve (amplify) each Fourier mode first 
c     2) compute inverse FFT on uniform fine grid
c     3) spread data to regular mesh using Gaussian
c
c
c     See subroutine nufft3d1f90
c     for more comments on fast gridding and parameter selection.
c
************************************************************************
c     Precision dependent parameters
c
c     rat is oversampling parameter
c     nspread is number of neighbors to which Gaussian gridding is
c     carried out.
c     -------------------------------
      ier = 0
      if (eps.lt.1d-33 .or. eps.gt.1d-1) then
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
      nf2 = rat*mt
      nf3 = rat*mu
      if (2*nspread.gt.min(nf1,nf2,nf3)) then
         ii = next235(2d0*nspread)
         nf1 = max(nf1, ii)
         nf2 = max(nf2, ii)
         nf3 = max(nf3, ii)
      endif
c
c     lambda (described above) = nspread/(rat*(rat-0.5d0))
c     It is more convenient to define r2lamb = rat*rat*lambda
c
      r2lamb = rat*rat * nspread / (rat*(rat-.5d0))
      hx = 2*pi/nf1
      hy = 2*pi/nf2
      hz = 2*pi/nf3
c
c     -------------------------------
c     Compute workspace size and allocate
c
c     nw1 for fine grid, nw2 for second/thrid index, nw3 for fft.
c     -------------------------------
      nw1 = nf1*nf2*nf3
      nw2 = nw1 + max(nf2,nf3)
      nw3 = nw2 + max(nf1,nf2,nf3)
      iw10 = 2*nw3
      iw11 = iw10 + ms
      iw12 = iw11 + mt
      iw13 = iw12 + mu/2 + 1
      iw14 = iw13 + 48+16+2*nf1
      iw15 = iw14 + 48+16+2*nf2
      iw16 = iw15 + 48+16+2*nf3
      allocate( fw(0:iw16-1) )
c
c     ---------------------------------------------------------------
c     Precompute spreading constants, initialize fw to hold one
c     of spreading terms for fast Gaussian gridding, initialize FFTs
c     ---------------------------------------------------------------
      t1 = pi/r2lamb
      t2 = pi/r2lamb
      t3 = pi/r2lamb
      do k1 = 1, nspread
         fw(iw13+k1) = exp(-t1*k1**2)
         fw(iw14+k1) = exp(-t2*k1**2)
         fw(iw15+k1) = exp(-t3*k1**2)
      enddo

      call dcfti1(nf1,fw(iw13+64),fw(iw13+48))
      call dcfti1(nf2,fw(iw14+64),fw(iw14+48))
      call dcfti1(nf3,fw(iw15+64),fw(iw15+48))
c
c     ---------------------------------------------------------------
c     Precompute deconvolution contants
c     ---------------------------------------------------------------
c
      t1 = pi * r2lamb / dble(nf1)**2
      cross1 = 1d0-2d0*mod(ms/2,2)
      do k1 = -ms/2, (ms-1)/2
         fw(iw10+k1+ms/2)=cross1*exp(t1*dble(k1)**2)
         cross1 = -cross1
      enddo
      t2 = pi * r2lamb / dble(nf2)**2
      cross1 = 1d0-2d0*mod(mt/2,2)
      do k1 = -mt/2, (mt-1)/2
         fw(iw11+k1+mt/2)=cross1*exp(t2*dble(k1)**2)
         cross1 = -cross1
      enddo
      t3 = pi * r2lamb / dble(nf3)**2
      cross1 = 1d0 / (sqrt(r2lamb)*r2lamb)
      do k1 = 0, mu/2
         fw(iw12+k1)=cross1*exp(t3*dble(k1)**2)
         cross1 = -cross1
      enddo
c
c     ---------------------------------------------------------------
c     Deconvolve and compute inverse 3D FFT
c     Factors of (-1)**k needed to shift phase.
c     ---------------------------------------------------------------
      i3 = iw15+48
      do k2 = -mt/2,  (mt-1)/2
         do k1 = -ms/2, (ms-1)/2
            zz = fw(iw12) * fk(k1,k2,0)
            fw(2*nw1) = dreal(zz)
            fw(2*nw1+1) = dimag(zz)
            do k3 = 1, (mu-1)/2
	       istart = 2*(nw1+k3)
               zz = fw(iw12+k3)*fk(k1,k2,k3)
               fw(istart) = dreal(zz)
               fw(istart+1) = dimag(zz)
	       istart = 2*(nw1+nf3-k3)
               zz = fw(iw12+k3) * fk(k1,k2,-k3)
               fw(istart) = dreal(zz)
               fw(istart+1) = dimag(zz)
            enddo
            if (mu/2*2.eq.mu) then
	       istart = 2*(nw1+nf3-mu/2)
               zz = fw(iw12+mu/2) * fk(k1,k2,-mu/2)
               fw(istart) = dreal(zz)
               fw(istart+1) = dimag(zz)
	    endif
            do k3 = (mu+1)/2, nf3-mu/2-1
	       istart = 2*(nw1+k3)
               fw(istart) = 0d0
               fw(istart+1) = 0d0
            enddo
c
            if (iflag .ge. 0) then
               call dcftb1(nf3,fw(2*nw1),fw(2*nw2),fw(i3+16),fw(i3))
            else
               call dcftf1(nf3,fw(2*nw1),fw(2*nw2),fw(i3+16),fw(i3))
            endif
c
            ii = k1
            if (k1.lt.0) ii = ii+nf1
            ii = ii + k2*nf1
            if (k2.lt.0) ii = ii+nf1*nf2
            cross = fw(iw10+k1+ms/2) * fw(iw11+k2+mt/2)
            do k3 = 0, nf3-1
	       istart = 2*(ii+k3*nf1*nf2)
	       is2 = 2*(nw1+k3)
               zz = dcmplx(fw(is2),fw(is2+1))
               c2 = cross*zz
               fw(istart) = dreal(c2)
               fw(istart+1) = dimag(c2)
            enddo
         enddo
      enddo
c
      i3 = iw14 + 48
      do k3 = 0, nf3-1
         do k1 = -ms/2, (ms-1)/2
            ii = k1
            if (k1.lt.0) ii = ii + nf1
            ii = ii + k3 * nf1*nf2  ! cfine(k1,0,k3) = cw(k1+k3*nf1*nf2)

            do k2 = 0, (mt-1)/2
	       istart = 2*(nw1+k2)
	       is2 = 2*(ii+k2*nf1)
               fw(istart) = fw(is2)
               fw(istart+1) = fw(is2+1)
            enddo
            do k2 = (mt+1)/2, nf2-mt/2-1
	       istart = 2*(nw1+k2)
               fw(istart) = 0d0
               fw(istart+1) = 0d0
            enddo
            do k2 = nf2-mt/2, nf2-1
	       istart = 2*(nw1+k2)
	       is2 = 2*(ii+k2*nf1)
               fw(istart) = fw(is2)
               fw(istart+1) = fw(is2+1)
            enddo
            if (iflag .ge. 0) then
               call dcftb1(nf2,fw(2*nw1),fw(2*nw2),fw(i3+16),fw(i3))
            else
               call dcftf1(nf2,fw(2*nw1),fw(2*nw2),fw(i3+16),fw(i3))
            endif
            do k2 = 0, nf2-1
	       istart = 2*(ii+k2*nf1)
	       is2 = 2*(nw1+k2)
               fw(istart) = fw(is2)
               fw(istart+1) = fw(is2+1)
            enddo
         enddo
      enddo
c
      i3 = iw13 + 48
      do k3 = 0, nf3-1
         do k2 = 0, nf2-1
            ii = nf1 * (k2+k3*nf2)  ! cfine(0,k2,k3) = cw(nf1*(k2+k3*nf2))
c
            do k1 = (ms+1)/2, nf1-ms/2-1
	       istart = 2*(ii+k1)
               fw(istart) = 0d0
               fw(istart+1) = 0d0
            enddo
            if (iflag .ge. 0) then
               call dcftb1(nf1,fw(2*ii),fw(2*nw2),fw(i3+16),fw(i3))
            else
               call dcftf1(nf1,fw(2*ii),fw(2*nw2),fw(i3+16),fw(i3))
            endif
         enddo
      enddo
c
c     ---------------------------------------------------------------
c     Loop over target points (1,...,nj)
c
c       1. find closest mesh point (with periodic wrapping if needed)
c       2. spread source data onto nearest nspread**3 grid points
c          using fast Gaussian gridding. 
c
c     Code is not easy to read because fast gridding is optimized to 
c     minimize the number of multiplies in inner loop and there is a
c     certain amout of unrolling involved.
c     ---------------------------------------------------------------
      t1 = pi/r2lamb
      t2 = pi/r2lamb
      t3 = pi/r2lamb
      do j = 1, nj
         cj(j) = dcmplx(0d0,0d0)
c
         jb1 = int((xj(j)+pi)/hx)
         diff1 = (xj(j)+pi)/hx - jb1
         jb1 = mod(jb1, nf1)
         if (jb1.lt.0) jb1=jb1+nf1
c
         jb2 = int((yj(j)+pi)/hy)
         diff2 = (yj(j)+pi)/hy - jb2
         jb2 = mod(jb2, nf2)
         if (jb2.lt.0) jb2=jb2+nf2
c
         jb3 = int((zj(j)+pi)/hz)
         diff3 = (zj(j)+pi)/hz - jb3
         jb3 = mod(jb3, nf3)
         if (jb3.lt.0) jb3=jb3+nf3
c
c        exp(-t1*(diff1-k1)**2) = xc(k1) / exp(-t1*(diff2**2+diff3**2))
c
         xc(0) = exp(-t1*(diff1**2+diff2**2+diff3**2))
         cross = xc(0)
         cross1 = exp(2d0*t1 * diff1)
         do k1 = 1, nspread
            cross = cross * cross1
            xc(k1) = fw(iw13+k1)*cross
         enddo
         cross = xc(0)
         cross1 = 1d0/cross1
         do k1 = 1, nspread-1
            cross = cross * cross1
            xc(-k1) = fw(iw13+k1)*cross
         enddo
c
c        exp(-t1*(diff2-k2)**2) = yc(k2) * exp(-t1*diff2**2)
c
         yc(0) = 1d0
         cross = exp(2d0*t2 * diff2)
         cross1 = cross
         do k2 = 1, nspread-1
            yc(k2) = fw(iw14+k2)*cross
            yc(-k2) = fw(iw14+k2)/cross
            cross = cross * cross1
         enddo
         yc(nspread) = fw(iw14+nspread)*cross
c
c        exp(-t1*(diff3-k3)**2) = zc(k3) * exp(-t1*diff3**2)
c
         zc(0) = 1d0
         cross = exp(2d0*t3 * diff3)
         cross1 = cross
         do k3 = 1, nspread-1
            zc(k3) = fw(iw15+k3)*cross
            zc(-k3) = fw(iw15+k3)/cross
            cross = cross * cross1
         enddo
         zc(nspread) = fw(iw15+nspread)*cross
c
         jb1d = min(nspread-1, jb1)
         jb1u = min(nspread, nf1-jb1-1)
         do k2 = -nspread+1, nspread
            i2(k2) = jb2+k2
            if (i2(k2).lt.0) then
               i2(k2) = i2(k2) + nf2
            elseif (i2(k2).ge.nf2) then
               i2(k2) = i2(k2) - nf2
            endif
         enddo
c
         do k3 = -nspread+1, nspread
            i3 = jb3+k3
            if (i3.lt.0) then
               i3 = i3 + nf3
            elseif (i3.ge.nf3) then
               i3 = i3 - nf3
            endif
c
            c2 = dcmplx(0d0, 0d0)
            do k2 = -nspread+1, nspread
               cc = dcmplx(0d0, 0d0)
               ii = jb1 + i2(k2)*nf1 + i3*nf1*nf2 ! cfine(ib,jb+k2,kb+k3)
               do k1 = -nspread+1, -jb1d-1
	          istart = 2*(k1+nf1+ii)
                  zz = dcmplx(fw(istart),fw(istart+1))
                  cc = cc + xc(k1)*zz
               enddo
               do k1 = -jb1d, jb1u
	          istart = 2*(k1+ii)
                  zz = dcmplx(fw(istart),fw(istart+1))
                  cc = cc + xc(k1)*zz
               enddo
               do k1 = jb1u+1, nspread
	          istart = 2*(k1-nf1+ii)
                  zz = dcmplx(fw(istart),fw(istart+1))
                  cc = cc + xc(k1)*zz
               enddo
               c2 = c2 + yc(k2)*cc
            enddo
            cj(j) = cj(j) + zc(k3)*c2
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
c***********************************************************************
      subroutine nufft3d3f90(nj,xj,yj,zj,cj,iflag,eps,
     1           nk,sk,tk,uk,fk,ier)
      implicit none
      integer ier,ii,iflag,i3
      integer iw7,iw8,iw9,iw10,iw11,iw12,iw13,iw14,iw15,iw16
      integer j,jb1,jb2,jb3,k1,k2,k3,kb1,kb2,kb3,next235
      integer nf1,nf2,nf3,nj,nk,nspread,nw1,nw2,nw3,istart,is2
      real*8 xj(nj),yj(nj),zj(nj),sk(nk),tk(nk),uk(nk)
      real*8 eps,pi,rat,t1,t2,t3,diff1,diff2,diff3
      real*8 rat1,rat2,rat3,r2lamb1,r2lamb2,r2lamb3
      real*8 xm,ym,sm,tm,zm,um,hx,hy,hz,hs,ht,hu
      real*8 ang,xb,yb,zb,sb,tb,ub
      real*8 cross, cross1,xc(-147:147),yc(-147:147),zc(-147:147)
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*16 cj(nj), fk(nk)
      complex*16 cc, c2, cs, zz
c ----------------------------------------------------------------------
      real*8, allocatable :: fw(:)
c ----------------------------------------------------------------------
c
c     if (iflag.ge.0) then 
c
c                nj
c       fk(k) = SUM cj(j) exp(+i sk(k) xj(j)) 
c               j=1       exp(+i tk(k) yj(j)) 
c                         exp(+i uk(k) zj(j))
c                                              for  k = 1, ..., nk
c     else
c                nj
c       fk(k) = SUM cj(j) exp(-i sk(k) xj(j)) 
c               j=1       exp(-i tk(k) yj(j)) 
c                         exp(-i uk(k) zj(j))
c                                              for  k = 1, ..., nk
c----------------------------------------------------------------------
c     INPUT:
c
c     nj       number of sources   (integer)
c     xj,yj,zj location of sources (real *8 arrays)
c     cj       strengths of sources (complex *16 array)
c     iflag    determines sign of FFT (see above)
c     eps      precision request  (between 1.0d-33 and 1.0d-1)
c               recomended value is 1d-15 for double precision calculations
c     nk       number of (noninteger) Fourier modes computed
c     sk,tk,uk k-values (locations) of desired Fourier modes
c
c     OUTPUT:
c
c     fk     Fourier transform values (double complex array)
c     ier      error return code
c
c              ier = 0 => normal execution
c              ier = 1 => precision requested is out of range.
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
************************************************************************
c
c     -------------------------------
c     check requested precision
c     -------------------------------
c
      ier = 0
      if ((eps.lt.1d-33).or.(eps.gt.1d-1)) then
         ier = 1
	 return
      endif
c
c     -------------------------------
c     Compute the ranges of {xj}/{sk} 
c     -------------------------------
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
      t1 = yj(1)
      t2 = yj(1)
      do j = 2, nj
         if (yj(j).gt.t2) then
             t2=yj(j)
         else if (yj(j).lt.t1) then
             t1=yj(j)
         endif
      enddo
      yb = (t1+t2) / 2d0
      ym = max(t2-yb,-t1+yb)
c
      t1 = tk(1)
      t2 = tk(1)
      do k1 = 2, nk
         if (tk(k1).gt.t2) then
             t2=tk(k1)
         else if (tk(k1).lt.t1) then
             t1=tk(k1)
         endif
      enddo
      tb = (t1+t2) / 2d0
      tm = max(t2-tb,-t1+tb)
c
      t1 = zj(1)
      t2 = zj(1)
      do j = 2, nj
         if (zj(j).gt.t2) then
             t2=zj(j)
         else if (zj(j).lt.t1) then
             t1=zj(j)
         endif
      enddo
      zb = (t1+t2) / 2d0
      zm = max(t2-zb,-t1+zb)
c
      t1 = uk(1)
      t2 = uk(1)
      do k1 = 2, nk
         if (uk(k1).gt.t2) then
             t2=uk(k1)
         else if (uk(k1).lt.t1) then
             t1=uk(k1)
         endif
      enddo
      ub = (t1+t2) / 2d0
      um = max(t2-ub,-t1+ub)
c
c     -------------------------------
c     Precision dependent parameters
c
c     rat is oversampling parameter
c     nspread is number ofneighbors to which Gaussian gridding is
c     carried out.
c     -------------------------------
      if (eps.le.1d-12) then
         rat = 3.0d0
      else if (eps.le.1d-11) then
         rat = sqrt(3.3d0)
      else
         rat = sqrt(2.2d0)
      endif
c
      nspread = int(-log(eps)/(pi*(rat-1d0)/(rat-.5d0)) + .5d0)
      t1 = 2d0/pi * xm*sm
      t2 = 2d0/pi * ym*tm
      t3 = 2d0/pi * zm*um
      nf1 = next235(rat*max(rat*t1+2*nspread,2*nspread/(rat-1)))
      nf2 = next235(rat*max(rat*t2+2*nspread,2*nspread/(rat-1)))
      nf3 = next235(rat*max(rat*t3+2*nspread,2*nspread/(rat-1)))
      rat1 = (sqrt(nf1*t1+nspread**2)-nspread)/t1
      rat2 = (sqrt(nf2*t2+nspread**2)-nspread)/t2
      rat3 = (sqrt(nf3*t3+nspread**2)-nspread)/t3
c
      r2lamb1 = rat1*rat1 * nspread / (rat1*(rat1-.5d0))
      r2lamb2 = rat2*rat2 * nspread / (rat2*(rat2-.5d0))
      r2lamb3 = rat3*rat3 * nspread / (rat3*(rat3-.5d0))
      hx = pi/(rat1*sm)
      hs = 2d0*pi/dble(nf1)/hx
      hy = pi/(rat2*tm)
      ht = 2d0*pi/dble(nf2)/hy
      hz = pi/(rat3*um)
      hu = 2d0*pi/dble(nf3)/hz
c
c     -------------------------------
c     Compute workspace size
c     -------------------------------
      nw1 = nf1*nf2*nf3
      nw2 = nw1 + max(nf2,nf3)
      nw3 = nw2 + max(nf1,nf2,nf3)
      iw7 = int(nf1*(r2lamb1-nspread)/r2lamb1+.1d0)
      iw8 = int(nf2*(r2lamb2-nspread)/r2lamb2+.1d0)
      iw9 = int(nf3*(r2lamb3-nspread)/r2lamb3+.1d0)
      iw10 = 2*nw3
      iw11 = iw10 + iw7+1
      iw12 = iw11 + iw8+1
      iw13 = iw12 + iw9+1
      iw14 = iw13 + 48+16+2*nf1
      iw15 = iw14 + 48+16+2*nf2
      iw16 = iw15 + 48+16+2*nf3
      allocate( fw(0:iw16-1) )
c
c     ---------------------------------------------------------------
c     Precompute spreading constants and Initialize rwork
c     ---------------------------------------------------------------
      t1 = pi/r2lamb1
      t2 = pi/r2lamb2
      t3 = pi/r2lamb3
      do k1 = 1, nspread
         fw(iw13+k1) = exp(-t1*k1**2)
         fw(iw14+k1) = exp(-t2*k1**2)
         fw(iw15+k1) = exp(-t3*k1**2)
      enddo
c
      call dcfti1(nf1,fw(iw13+64),fw(iw13+48))
      call dcfti1(nf2,fw(iw14+64),fw(iw14+48))
      call dcfti1(nf3,fw(iw15+64),fw(iw15+48))
c
c     ---------------------------------------------------------------
c     Precompute deconvolution contants
c     ---------------------------------------------------------------
c
      t1 = pi * r2lamb1 / dble(nf1)**2
      cross1 = (1d0-2d0*mod(nf1/2,2))/r2lamb1
      do k1 = 0, iw7
         fw(iw10+k1) = cross1*exp(t1*dble(k1)**2)
         cross1 = -cross1
      enddo
      t2 = pi * r2lamb2 / dble(nf2)**2
      cross1 = 1d0/r2lamb2
      do k1 = 0, iw8
         fw(iw11+k1) = cross1*exp(t2*dble(k1)**2)
         cross1 = -cross1
      enddo
      t3 = pi * r2lamb3 / dble(nf3)**2
      cross1 = 1d0/r2lamb3
      do k1 = 0, iw9
         fw(iw12+k1) = cross1*exp(t3*dble(k1)**2)
         cross1 = -cross1
      enddo
c
c     ---------------------------------------------------------------
c     Initialize fine grid data to zero.
c     ---------------------------------------------------------------
c
      do k1 = 0, 2*nf1*nf2*nf3-1
         fw(k1) = 0d0
      enddo
c
c     ---------------------------------------------------------------
c     Step 1/5 : gridding as in type 1 transform.
c ---------------------------------------------------------------
c
      t1 = pi/r2lamb1
      t2 = pi/r2lamb2
      t3 = pi/r2lamb3
      if (iflag .lt. 0) sb = -sb
      if (iflag .lt. 0) tb = -tb
      if (iflag .lt. 0) ub = -ub
      do j = 1, nj
         jb1 = int(dble(nf1/2) + (xj(j)-xb)/hx)
         diff1 = dble(nf1/2) + (xj(j)-xb)/hx - jb1
         jb2 = int(dble(nf2/2) + (yj(j)-yb)/hy)
         diff2 = dble(nf2/2) + (yj(j)-yb)/hy - jb2
         jb3 = int(dble(nf3/2) + (zj(j)-zb)/hz)
         diff3 = dble(nf3/2) + (zj(j)-zb)/hz - jb3
         ang = sb*xj(j) + tb*yj(j) + ub*zj(j)
         cs = dcmplx(cos(ang),sin(ang)) * cj(j)
c
c     exp(-t1*(diff1-k1)**2) = xc(k1) / exp(-t2*diff2**2-t3*diff3**2)
c
         xc(0) = exp(-t1*diff1**2-t2*diff2**2-t3*diff3**2)
         cross = xc(0)
         cross1 = exp(2d0*t1 * diff1)
         do k1 = 1, nspread
            cross = cross * cross1
            xc(k1) = fw(iw13+k1)*cross
         enddo
         cross = xc(0)
         cross1 = 1d0/cross1
         do k1 = 1, nspread-1
            cross = cross * cross1
            xc(-k1) = fw(iw13+k1)*cross
         enddo
c
c     exp(-t2*(diff2-k2)**2) = yc(k2) * exp(-t2*diff2**2)
c
         yc(0) = 1d0
         cross = exp(2d0*t2 * diff2)
         cross1 = cross
         do k2 = 1, nspread-1
            yc(k2) = fw(iw14+k2)*cross
            yc(-k2) = fw(iw14+k2)/cross
            cross = cross * cross1
         enddo
         yc(nspread) = fw(iw14+nspread)*cross
c
c     exp(-t3*(diff3-k3)**2) = zc(k3) * exp(-t3*diff3**2)
c
         zc(0) = 1d0
         cross = exp(2d0*t3 * diff3)
         cross1 = cross
         do k3 = 1, nspread-1
            zc(k3) = fw(iw15+k3)*cross
            zc(-k3) = fw(iw15+k3)/cross
            cross = cross * cross1
         enddo
         zc(nspread) = fw(iw15+nspread)*cross
c
         do k3 = -nspread+1, nspread
            c2 = zc(k3)*cs
            do k2 = -nspread+1, nspread
               cc = yc(k2)*c2
               ii = jb1 + (jb2+k2)*nf1 + (jb3+k3)*nf1*nf2
               do k1 = -nspread+1, nspread
                  istart = 2*(ii+k1)
                  zz = xc(k1)*cc
                  fw(istart) = fw(istart) + dreal(zz) 
                  fw(istart+1) = fw(istart+1) + dimag(zz) 
               enddo
            enddo
         enddo
      enddo
      if (iflag .lt. 0) sb = -sb
      if (iflag .lt. 0) tb = -tb
      if (iflag .lt. 0) ub = -ub
c
c     ---------------------------------------------------------------
c     Step 2/5 : Deconvolve (amplify) as in type 2 transform.
c     Step 3/5 : Compute 3D FFT with shift
c       (-1)^k F_(k+M/2) = Sum (-1)^j F_(j+M/2) e(2pi ijk/M)
c     ---------------------------------------------------------------
      i3 = iw15+48
      do k2 = -iw8, iw8
         do k1 = -iw7, iw7
            ii = (nf1/2+k1) + (nf2/2+k2)*nf1 + (nf3/2)*nf1*nf2
            cross = fw(iw10+abs(k1)) * fw(iw11+abs(k2))
	    istart = 2*nw1
	    c2 = dcmplx(fw(2*ii),fw(2*ii+1))
            zz = (cross*fw(iw12))*c2
            fw(istart) = dreal(zz)
            fw(istart+1) = dimag(zz)
            do k3 = 1, iw9
	       istart = 2*(nw1+k3)
	       is2 = 2*(ii+k3*nf1*nf2)
	       c2 = dcmplx(fw(is2),fw(is2+1))
               zz = (cross*fw(iw12+k3))*c2
               fw(istart) = dreal(zz)
               fw(istart+1) = dimag(zz)
	       istart = 2*(nw1+nf3-k3)
	       is2 = 2*(ii-k3*nf1*nf2)
	       c2 = dcmplx(fw(is2),fw(is2+1))
               zz = (cross*fw(iw12+k3))*c2
               fw(istart) = dreal(zz)
               fw(istart+1) = dimag(zz)
            enddo
            do k3 = iw9+1, nf3-iw9-1
	       istart = 2*(nw1+k3)
               fw(istart) = 0d0
               fw(istart+1) = 0d0
            enddo
c
            if (iflag .ge. 0) then
               call dcftb1(nf3,fw(2*nw1),fw(2*nw2),fw(i3+16),fw(i3))
            else
               call dcftf1(nf3,fw(2*nw1),fw(2*nw2),fw(i3+16),fw(i3))
            endif
c
            do k3 = -iw9-nspread, iw9+nspread
	       istart = 2*(ii+k3*nf1*nf2)
	       is2 = 2*(nw1+nf3/2+k3)
               fw(istart) = fw(is2)
               fw(istart+1) = fw(is2+1)
            enddo
         enddo
      enddo
c
      i3 = iw14 + 48
      do k3 = -iw9-nspread, iw9+nspread
         do k1 = -iw7, iw7
            ii = (nf1/2+k1) + (nf2/2)*nf1 + (nf3/2+k3)*nf1*nf2
	    istart = 2*nw1
	    is2 = 2*ii
            fw(istart) = fw(is2)
            fw(istart+1) = fw(is2+1)
            do k2 = 0, iw8
	       istart = 2*(nw1+k2)
	       is2 = 2*(ii+k2*nf1)
               fw(istart) = fw(is2)
               fw(istart+1) = fw(is2+1)
	       istart = 2*(nw1+nf2-k2)
	       is2 = 2*(ii-k2*nf1)
               fw(istart) = fw(is2)
               fw(istart+1) = fw(is2+1)
            enddo
            do k2 = iw8+1, nf2-iw8-1
	       istart = 2*(nw1+k2)
               fw(istart) = 0d0
               fw(istart+1) = 0d0
            enddo
            if (iflag .ge. 0) then
               call dcftb1(nf2,fw(2*nw1),fw(2*nw2),fw(i3+16),fw(i3))
            else
               call dcftf1(nf2,fw(2*nw1),fw(2*nw2),fw(i3+16),fw(i3))
            endif
            do k2 = -iw8-nspread, iw8+nspread
	       istart = 2*(ii+k2*nf1)
	       is2 = 2*(nw1+nf2/2+k2)
               fw(istart) = fw(is2)
               fw(istart+1) = fw(is2+1)
            enddo
         enddo
      enddo
c
      i3 = iw13 + 48
      do k3 = -iw9-nspread, iw9+nspread
         do k2 = -iw8-nspread, +iw8+nspread
            ii = (nf2/2+k2)*nf1 + (nf3/2+k3)*nf1*nf2
            if (iflag .ge. 0) then
               call dcftb1(nf1,fw(2*ii),fw(2*nw2),fw(i3+16),fw(i3))
            else
               call dcftf1(nf1,fw(2*ii),fw(2*nw2),fw(i3+16),fw(i3))
            endif
            do k1 = 1, iw7+nspread, 2
	       istart = 2*(ii+nf1/2+k1)
	       is2 = 2*(ii+nf1/2-k1)
               fw(istart) = -fw(istart)
               fw(istart+1) = -fw(istart+1)
               fw(is2) = -fw(is2)
               fw(is2+1) = -fw(is2+1)
            enddo
         enddo
      enddo
c
c     ---------------------------------------------------------------
c     Step 4/5 : Gaussian gridding to irregular points.
c     Step 5/5 : Final deconvolution.
c     ---------------------------------------------------------------
c
      t1 = pi/r2lamb1
      t2 = pi/r2lamb2
      t3 = pi/r2lamb3
      do j = 1, nk
         kb1 = int(dble(nf1/2) + (sk(j)-sb)/hs)
         diff1 = dble(nf1/2) + (sk(j)-sb)/hs - kb1
         kb2 = int(dble(nf2/2) + (tk(j)-tb)/ht)
         diff2 = dble(nf2/2) + (tk(j)-tb)/ht - kb2
         kb3 = int(dble(nf3/2) + (uk(j)-ub)/hu)
         diff3 = dble(nf3/2) + (uk(j)-ub)/hu - kb3
c
c     exp(-t1*(diff1-k1)**2) = xc(k1) / exp(-t2*diff2**2-t3*diff3**2)
c
         xc(0) = exp(-t1*diff1**2-t2*diff2**2-t3*diff3**2)
         cross = xc(0)
         cross1 = exp(2d0*t1 * diff1)
         do k1 = 1, nspread
            cross = cross * cross1
            xc(k1) = fw(iw13+k1)*cross
         enddo
         cross = xc(0)
         cross1 = 1d0/cross1
         do k1 = 1, nspread-1
            cross = cross * cross1
            xc(-k1) = fw(iw13+k1)*cross
         enddo
c
c     exp(-t2*(diff2-k2)**2) = yc(k2) * exp(-t2*diff2**2)
c
         yc(0) = 1d0
         cross = exp(2d0*t2 * diff2)
         cross1 = cross
         do k2 = 1, nspread-1
            yc(k2) = fw(iw14+k2)*cross
            yc(-k2) = fw(iw14+k2)/cross
            cross = cross * cross1
         enddo
         yc(nspread) = fw(iw14+nspread)*cross
c
c     exp(-t3*(diff3-k3)**2) = zc(k3) * exp(-t3*diff3**2)
c
         zc(0) = 1d0
         cross = exp(2d0*t3 * diff3)
         cross1 = cross
         do k3 = 1, nspread-1
            zc(k3) = fw(iw15+k3)*cross
            zc(-k3) = fw(iw15+k3)/cross
            cross = cross * cross1
         enddo
         zc(nspread) = fw(iw15+nspread)*cross
c
         fk(j) = dcmplx(0d0,0d0)
         do k3 = -nspread+1, nspread
            do k2 = -nspread+1, nspread
               ii = kb1 + (kb2+k2)*nf1 + (kb3+k3)*nf1*nf2
               cross = yc(k2)*zc(k3)
               do k1 = -nspread+1, nspread
		  is2 = 2*(ii+k1)
		  zz = dcmplx(fw(is2),fw(is2+1))
                  fk(j) = fk(j) + (xc(k1)*cross)*zz
               enddo
            enddo
         enddo
      enddo
c
      t1 = r2lamb1/(4d0*pi) * hx**2
      t2 = r2lamb2/(4d0*pi) * hy**2
      t3 = r2lamb3/(4d0*pi) * hz**2
      if (iflag .lt. 0) xb = -xb
      if (iflag .lt. 0) yb = -yb
      if (iflag .lt. 0) zb = -zb
      do j = 1, nk
         fk(j) = (exp(t1*(sk(j)-sb)**2
     &                  +t2*(tk(j)-tb)**2+t3*(uk(j)-ub)**2))*fk(j)
         ang = (sk(j)-sb)*xb + (tk(j)-tb)*yb + (uk(j)-ub)*zb
         fk(j) = dcmplx(cos(ang),sin(ang)) * fk(j)
      enddo
      deallocate(fw)
      return
      end
