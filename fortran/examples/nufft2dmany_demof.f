cc Copyright (C) 2004-2009: Leslie Greengard and June-Yub Lee 
cc Contact: greengard@cims.nyu.edu
cc 
cc This software is being released under a FreeBSD license
cc (see license.txt in this directory). 
cc
c tweaked Alex Barnett to call FINUFFT 2/17/17
c dyn malloc; type 2 uses same input data fk0, other bugs 3/8/17
c     Single-prec version 9/24/18
c
c Compile with (multithreaded version):
c     gfortran nufft2dmany_demof.f dirft2df.f -o nufft2dmany_demof
c      ../lib/libfinufft.a -lstdc++ -lfftw3 -lfftw3_omp -lm -fopenmp
c
      program nufft2dmany_demo

      implicit none
c
      integer i,ier,iflag,j,k1,k2,mx,ms,mt,n1,n2,nj,nk,ndata,d
      real*4, allocatable :: xj(:),yj(:),sk(:),tk(:)
      real*4 err,pi,eps,salg,ealg,maxerr
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*8, allocatable :: cj(:),cj0(:),cj1(:),fk0(:),fk1(:)
c
c     --------------------------------------------------
c     create some test data
c     --------------------------------------------------
c
      ndata = 2
      n1 = 36
      n2 = 40
      ms = 32
      mt = 30
      nj = n1*n2
      nk = ms*mt

      maxerr = 0.0
c     first alloc everything
      allocate(xj(nj))
      allocate(yj(nj))
      allocate(fk0(nk*ndata))
      allocate(fk1(nk*ndata))
      allocate(cj (nj*ndata))
      allocate(cj0(nj*ndata))
      allocate(cj1(nj*ndata))
      do k1 = -n1/2, (n1-1)/2
         do k2 = -n2/2, (n2-1)/2
            j = (k2+n2/2+1) + (k1+n1/2)*n2
            xj(j) = pi*cos(-pi*k1/n1)
            yj(j) = pi*cos(-pi*k2/n2)
            do d = 0, ndata-1
                cj(j+d*nj) = cmplx(sin(pi*j/n1+d),cos(pi*j/n2+d))
            enddo
         enddo
      enddo
c
c     -----------------------
c     start tests
c     -----------------------
c
      iflag = 1
      print*,'Starting 2Dmany testing: ', ' nj =',nj, ' ms,mt =',ms,mt
      do i = 1,3
         if (i.eq.1) eps=1d-4
         if (i.eq.2) eps=1d-8
         if (i.eq.3) eps=1d-12
	 print*,' '
	 print*,' Requested precision eps =',eps
	 print*,' '
c
c     -----------------------
c     call 2D Type 1 method
c     -----------------------
c
         call finufft2d1many_f(ndata,nj,xj,yj,cj,iflag, 
     &                         eps,ms,mt,fk1,ier)
         do d = 1, ndata
            call dirft2d1f(nj,xj,yj,cj(1+(d-1)*nj:d*nj),iflag,ms,mt,
     &                    fk0(1+(d-1)*nk:d*nk))
            call errcomp(fk0(1+(d-1)*nk:d*nk),fk1(1+(d-1)*nk:d*nk),
     &                   nk,err)
            maxerr = max(maxerr,err)
         enddo
         print *, ' max type 1 err = ',err
c
c     -----------------------
c      call 2D Type 2 method
c     -----------------------
         call finufft2d2many_f(ndata,nj,xj,yj,cj1,iflag,
     &                         eps,ms,mt,fk0,ier)
         do d = 1, ndata
            call dirft2d2f(nj,xj,yj,cj0(1+(d-1)*nj:d*nj),iflag,ms,mt,
     &                    fk0(1+(d-1)*nk:d*nk))
            call errcomp(cj0(1+(d-1)*nj:d*nj),cj1(1+(d-1)*nj:d*nj),
     &                   nj,err)
            maxerr = max(maxerr,err)
         enddo

         print *, ' max type 2 err = ',err

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
