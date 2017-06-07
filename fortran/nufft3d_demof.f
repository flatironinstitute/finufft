cc Copyright (C) 2004-2009: Leslie Greengard and June-Yub Lee 
cc Contact: greengard@cims.nyu.edu
cc 
cc This software is being released under a FreeBSD license
cc (see license.txt in this directory). 
cc
c tweaked Alex Barnett to call FINUFFT 2/17/17
c dyn malloc; type 2 uses same input data fk0, other bugs 3/8/17
c Single-prec version, 4/5/17     
c
c Compile with (multithreaded version):
c gfortran nufft3d_demof.f dirft3df.f -o nufft3d_demof ../lib/libfinufft.a
c          -lstdc++ -lfftw3f -lfftw3f_omp -lm -fopenmp
c
      program nufft3d_demof
      implicit none
c
      integer i,ier,iflag,j,k1,k2,k3,mx,ms,mt,mu,n1,n2,n3,nj,nk
      real*4, allocatable :: xj(:),yj(:),zj(:),sk(:),tk(:),uk(:)
      real*4 err,pi,eps,salg,ealg
      real*4 t0,t1,second
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*8, allocatable :: cj(:),cj0(:),cj1(:),fk0(:),fk1(:)
c
c     --------------------------------------------------
c     create some test data
c     --------------------------------------------------
c
      ms = 24/2
      mt = 16/2
      mu = 18/2
      n1 = 16/2
      n2 = 18/2
      n3 = 24/2
      nj = n1*n2*n3
      nk = ms*mt*mu
c     first alloc everything
      allocate(fk0(nk))
      allocate(fk1(nk))
      allocate(sk(nk))
      allocate(tk(nk))
      allocate(uk(nk))
      allocate(xj(nj))
      allocate(yj(nj))
      allocate(zj(nj))
      allocate(cj(nj))
      allocate(cj0(nj))
      allocate(cj1(nj))
      do k3 = -n3/2, (n3-1)/2
         do k2 = -n2/2, (n2-1)/2
            do k1 = -n1/2, (n1-1)/2
               j =  (k1+n1/2+1) + (k2+n2/2)*n1 + (k3+n3/2)*n1*n2
               xj(j) = pi*cos(-pi*k1/n1)
               yj(j) = pi*cos(-pi*k2/n2)
               zj(j) = pi*cos(-pi*k3/n3)
               cj(j) = dcmplx(sin(pi*j/n1),cos(pi*j/n2))
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
      do i = 1,3
         if (i.eq.1) eps=1d-2
         if (i.eq.2) eps=1d-4
         if (i.eq.3) eps=1d-6
	 print*,' '
	 print*,' Requested precision eps =',eps
	 print*,' '
c
c     -----------------------
c     call 3D Type 1 method
c     -----------------------
c
         call dirft3d1f(nj,xj,yj,zj,cj,iflag,ms,mt,mu,fk0)
         call finufft3d1_f(nj,xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk1,ier)
         print *, ' ier = ',ier
         call errcomp(fk0,fk1,nk,err)
         print *, ' type 1 err = ',err
c
c     -----------------------
c      call 3D Type 2 method
c     -----------------------
         call dirft3d2f(nj,xj,yj,zj,cj0,iflag,ms,mt,mu,fk0)
         call finufft3d2_f(nj,xj,yj,zj,cj1,iflag,eps,ms,mt,mu,fk0,ier)
         print *, ' ier = ',ier
         call errcomp(cj0,cj1,nj,err)
         print *, ' type 2 err = ',err
c
c     -----------------------
c      call 3D Type3 method
c     -----------------------
         do k1 = 1, nk
            sk(k1) = 12*(cos(k1*pi/nk))
            tk(k1) = 8*(sin(-pi/2+k1*pi/nk))
            uk(k1) = 10*(cos(k1*pi/nk))
         enddo

         call dirft3d3f(nj,xj,yj,zj,cj,iflag,nk,sk,tk,uk,fk0)
         call finufft3d3_f(nj,xj,yj,zj,cj,iflag,eps,nk,sk,tk,uk,fk1,ier)
         print *, ' ier = ',ier
         call errcomp(fk0,fk1,nk,err)
         print *, ' type 3 err = ',err
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
      complex*8 fk0(n), fk1(n)
      real *4 salg,ealg,err
c
      ealg = 0d0
      salg = 0d0
      do k = 1, n
         ealg = ealg + cabs(fk1(k)-fk0(k))**2
         salg = salg + cabs(fk0(k))**2
      enddo
      err =sqrt(ealg/salg)
      return
      end
