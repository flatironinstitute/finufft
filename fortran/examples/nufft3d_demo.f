c     Demo using FINUFFT for double-precision 3d transforms in legacy fortran.
c     Does types 1,2,3, including math test against direct summation.
c     Default opts only (see simple1d1 for how to change opts).
c
c     A slight modification of drivers from the CMCL NUFFT, (C) 2004-2009,
c     Leslie Greengard and June-Yub Lee. See: cmcl_license.txt.
c
c     Tweaked by Alex Barnett to call FINUFFT 2/17/17.
c     dyn malloc; type 2 uses same input data fk0, 3/8/17
c     Also see: ../README.
c
c     Compile with, eg (GCC, multithreaded, static, paste to a single line):
c
c     gfortran nufft3d_demo.f ../directft/dirft3d.f -o nufft3d_demo
c     ../../lib-static/libfinufft.a -lstdc++ -lfftw3 -lfftw3_omp -lm -fopenmp
c
      program nufft3d_demo
      implicit none

c     our fortran-header, always needed
      include 'finufft.fh'

      integer i,ier,iflag,j,k1,k2,k3,mx,n1,n2,n3
      integer*8 ms,mt,mu,nj,nk
      real*8, allocatable :: xj(:),yj(:),zj(:),sk(:),tk(:),uk(:)
      real*8 err,pi,eps,salg,ealg
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*16, allocatable :: cj(:),cj0(:),cj1(:),fk0(:),fk1(:)
c     for default opts, make a null pointer...
      type(finufft_opts), pointer :: defopts => null()
      
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
      do i = 1,4
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
         call dirft3d1(nj,xj,yj,zj,cj,iflag,ms,mt,mu,fk0)
         call finufft3d1(nj,xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk1,defopts,
     1        ier)
         print *, ' ier = ',ier
         call errcomp(fk0,fk1,nk,err)
         print *, ' type 1 error = ',err
c
c     -----------------------
c      call 3D Type 2 method
c     -----------------------
         call dirft3d2(nj,xj,yj,zj,cj0,iflag,ms,mt,mu,fk0)
         call finufft3d2(nj,xj,yj,zj,cj1,iflag,eps,ms,mt,mu,fk0,defopts,
     1        ier)
         print *, ' ier = ',ier
         call errcomp(cj0,cj1,nj,err)
         print *, ' type 2 error = ',err
c
c     -----------------------
c      call 3D Type3 method
c     -----------------------
         do k1 = 1, nk
            sk(k1) = 12*(dcos(k1*pi/nk))
            tk(k1) = 8*(dsin(-pi/2+k1*pi/nk))
            uk(k1) = 10*(dcos(k1*pi/nk))
         enddo

         call dirft3d3(nj,xj,yj,zj,cj,iflag,nk,sk,tk,uk,fk0)
         call finufft3d3(nj,xj,yj,zj,cj,iflag,eps,nk,sk,tk,uk,fk1,
     1        defopts,ier)
         print *, ' ier = ',ier
         call errcomp(fk0,fk1,nk,err)
         print *, ' type 3 error = ',err
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
      integer*8 k,n
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
