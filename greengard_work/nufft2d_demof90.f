cc Copyright (C) 2004-2009: Leslie Greengard and June-Yub Lee 
cc Contact: greengard@cims.nyu.edu
cc 
cc This software is being released under a FreeBSD license
cc (see license.txt in this directory). 
cc
      program testfft
      implicit none
c
      integer i,ier,iflag,j,k1,k2,mx,ms,mt,n1,n2,nj,nk
      parameter (mx=256*256)
      real*8 xj(mx),yj(mx)
      real *8 sk(mx),tk(mx)
      real*8 err,pi,eps,salg,ealg
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*16 cj(mx),cj0(mx),cj1(mx)
      complex*16 fk0(mx),fk1(mx)
c
c     --------------------------------------------------
c     create some test data
c     --------------------------------------------------
c
      n1 = 34
      n2 = 42
      ms = 38
      mt = 44
      nj = n1*n2
      do k1 = -n1/2, (n1-1)/2
         do k2 = -n2/2, (n2-1)/2
            j = (k2+n2/2+1) + (k1+n1/2)*n2
            xj(j) = pi*dcos(-pi*k1/n1)
            yj(j) = pi*dcos(-pi*k2/n2)
            cj(j) = dcmplx(dsin(pi*j/n1),dcos(pi*j/n2))
         enddo
      enddo
ccc      nj = 1
ccc      xj(1) = 0.0d0
ccc      yj(1) = 0.0d0
ccc      cj(1) = 1.0d0
c
c     -----------------------
c     start tests
c     -----------------------
c
      iflag = 1
      print*,'Starting 2D testing: ', ' nj =',nj, ' ms,mt =',ms,mt
      do i = 2,2
         if (i.eq.1) eps=1d-4
         if (i.eq.2) eps=1d-8
         if (i.eq.3) eps=1d-12
         if (i.eq.4) eps=1d-16
c extented/quad precision tests
         if (i.eq.5) eps=1d-20
         if (i.eq.6) eps=1d-24
         if (i.eq.7) eps=1d-28
         if (i.eq.8) eps=1d-32
	 print*,' '
	 print*,' Requested precision eps =',eps
	 print*,' '
c
c     -----------------------
c     call 2D Type 1 method
c     -----------------------
c
         call dirft2d1(nj,xj,yj,cj,iflag,ms,mt,fk0)
ccc         call nufft2d1f90(nj,xj,yj,cj,iflag,eps,ms,mt,fk1,ier)
         call nufft2d1f90x(nj,xj,yj,cj,iflag,eps,ms,mt,fk1,ier)
         call errcomp(fk0,fk1,ms*mt,err)
         print *, ' fk0 = ',fk0(1)
         print *, ' fk1 = ',fk1(1)
         print *, ' ier = ',ier
         call errcomp(fk0,fk1,ms*mt,err)
         print *, ' type 1 err = ',err
c
      enddo 
      stop
      end
c
c
c
c
c
      subroutine errcomp(fk0,fk1,n,err)
      implicit none
      integer k,n
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
