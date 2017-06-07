cc Copyright (C) 2004-2009: Leslie Greengard and June-Yub Lee 
cc Contact: greengard@cims.nyu.edu
cc 
cc This software is being released under a FreeBSD license
cc (see license.txt in this directory). 
cc
c tweaked Alex Barnett to call FINUFFT 2/17/17
c dyn malloc; type 2 uses same input data fk0, other bugs 3/8/17
c Single-prec version 4/5/17
c      
c Compile with (multithreaded version):
c gfortran nufft2d_demof.f dirft2df.f -o nufft2d_demof ../lib/libfinufft.a
c          -lstdc++ -lfftw3f -lfftw3f_threads -lm -fopenmp
c
      program nufft2d_demof
      implicit none
c
      integer i,ier,iflag,j,k1,k2,mx,ms,mt,n1,n2,nj,nk
      real*4, allocatable :: xj(:),yj(:),sk(:),tk(:)
      real*4 err,pi,eps,salg,ealg
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*8, allocatable :: cj(:),cj0(:),cj1(:),fk0(:),fk1(:)
c
c     --------------------------------------------------
c     create some test data
c     --------------------------------------------------
c
      n1 = 36
      n2 = 40
      ms = 32
      mt = 30
      nj = n1*n2
      nk = ms*mt
c     first alloc everything
      allocate(fk0(nk))
      allocate(fk1(nk))
      allocate(sk(nk))
      allocate(tk(nk))
      allocate(xj(nj))
      allocate(yj(nj))
      allocate(cj(nj))
      allocate(cj0(nj))
      allocate(cj1(nj))
      do k1 = -n1/2, (n1-1)/2
         do k2 = -n2/2, (n2-1)/2
            j = (k2+n2/2+1) + (k1+n1/2)*n2
            xj(j) = pi*cos(-pi*k1/n1)
            yj(j) = pi*cos(-pi*k2/n2)
            cj(j) = cmplx(sin(pi*j/n1),cos(pi*j/n2))
         enddo
      enddo
c
c     -----------------------
c     start tests
c     -----------------------
c
      iflag = 1
      print*,'Starting 2D testing: ', ' nj =',nj, ' ms,mt =',ms,mt
      do i = 1,3
         if (i.eq.1) eps=1d-2
         if (i.eq.2) eps=1d-4
         if (i.eq.3) eps=1d-6
	 print*,' '
	 print*,' Requested precision eps =',eps
	 print*,' '
c
c     -----------------------
c     call 2D Type 1 method
c     -----------------------
c
         call dirft2d1f(nj,xj,yj,cj,iflag,ms,mt,fk0)
         call finufft2d1_f(nj,xj,yj,cj,iflag,eps,ms,mt,fk1,ier)
         call errcomp(fk0,fk1,nk,err)
         print *, ' ier = ',ier
         call errcomp(fk0,fk1,nk,err)
         print *, ' type 1 err = ',err
c
c     -----------------------
c      call 2D Type 2 method
c     -----------------------
         call dirft2d2f(nj,xj,yj,cj0,iflag,ms,mt,fk0)
         call finufft2d2_f(nj,xj,yj,cj1,iflag,eps,ms,mt,fk0,ier)
         print *, ' ier = ',ier
         call errcomp(cj0,cj1,nj,err)
         print *, ' type 2 err = ',err
c
c     -----------------------
c      call 2D Type3 method
c     -----------------------
         do k1 = 1, nk
            sk(k1) = 48*(cos(k1*pi/nk))
            tk(k1) = 32*(sin(-pi/2+k1*pi/nk))
         enddo

         call dirft2d3f(nj,xj,yj,cj,iflag,nk,sk,tk,fk0)
         call finufft2d3_f(nj,xj,yj,cj,iflag,eps,nk,sk,tk,fk1,ier)
c
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
