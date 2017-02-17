cc Copyright (C) 2004-2009: Leslie Greengard and June-Yub Lee 
cc Contact: greengard@cims.nyu.edu
cc 
cc This software is being released under a FreeBSD license
cc (see license.txt in this directory). 
cc
      program testfft
      implicit none
c
      integer i,ier,iflag,j,k1,k2,k3,mx,ms,mt,mu,n1,n2,n3,nj,nk
      parameter (mx=1000 000)
      real*8 xj(mx),yj(mx),zj(mx)
      real *8 sk(mx),tk(mx),uk(mx)
      real*8 err,pi,eps,salg,ealg
      real*8 t2,t3,t4
      real*8 t0,t1,second
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*16 cj(mx),cj0(mx),cj1(mx)
      complex*16 fk0(mx),fk1(mx)
c
c     --------------------------------------------------
c     create some test data
c     --------------------------------------------------
c
      ms = 24
      mt = 16
      mu = 18
      n1 = 16
      n2 = 18
      n3 = 24
      nj = n1*n2*n3
      do k3 = -n3/2, (n3-1)/2
         do k2 = -n2/2, (n2-1)/2
            do k1 = -n1/2, (n1-1)/2
               j =  1 + (k1+n1/2+1) + (k2+n2/2)*n1 + (k3+n3/2)*n1*n2
               xj(j) = pi*dcos(-pi*k1/n1)
               yj(j) = pi*dcos(-pi*k2/n2)
               zj(j) = pi*dcos(-pi*k3/n3)
               cj(j) = dcmplx(dsin(pi*j/n1),dcos(pi*j/n2))
            enddo
         enddo
      enddo
c
c     -----------------------
c     start tests
c     -----------------------
c
      iflag = 1
      print*,'Starting 3D testing: ', 'nj =',nj, 'ms,mt,mu =',ms,mt,mu
      do i = 2,2
         if (i.eq.1) eps=1d-4
         if (i.eq.2) eps=1d-8
         if (i.eq.3) eps=1d-12
         if (i.eq.4) eps=1d-16
c extended/quad precision tests
         if (i.eq.5) eps=1d-20
         if (i.eq.6) eps=1d-24
         if (i.eq.7) eps=1d-28
         if (i.eq.8) eps=1d-32
	 print*,' '
	 print*,' Requested precision eps =',eps
	 print*,' '
c
c     -----------------------
c     call 3D Type 1 method
c     -----------------------
c
       t1 = second()
         call dirft3d1(nj,xj,yj,zj,cj,iflag,ms,mt,mu,fk0)
       t2 = second()
        call finufft3d1(nj,xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk1,ier)
       t3 = second()
        call nufft3d1f90(nj,xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk1,ier)
       t4 = second()
       write(6,*)' dir t2-t1 ',t2-t1
       write(6,*)' new t3-t2 ',t3-t2
       write(6,*)' old t4-t3 ',t4-t3
         print *, ' ier = ',ier
         call errcomp(fk0,fk1,ms*mt*mu,err)
         print *, ' type 1 err = ',err
c
c     -----------------------
c      call 3D Type 2 method
c     -----------------------
         call dirft3d2(nj,xj,yj,zj,cj0,iflag,ms,mt,mu,fk0)
         call finufft3d2(nj,xj,yj,zj,cj1,iflag,eps,ms,mt,mu,fk1,ier)
         print *, ' ier = ',ier
         call errcomp(cj0,cj1,nj,err)
         print *, ' type 2 err = ',err
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
