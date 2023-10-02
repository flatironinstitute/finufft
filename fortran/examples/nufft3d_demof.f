c     Demo using FINUFFT for single-precision 3d transforms in legacy fortran.
c     Does types 1,2,3, including math test against direct summation.
c     Default opts only (see simple1d1f for how to change opts).
c
c     A slight modification of drivers from the CMCL NUFFT, (C) 2004-2009,
c     Leslie Greengard and June-Yub Lee. See: cmcl_license.txt.
c
c     Tweaked by Alex Barnett to call FINUFFT 2/17/17, single-prec.
c     dyn malloc; type 2 uses same input data fk0, 3/8/17
c     Also see: ../README.
c
c     Compile with, eg (GCC, multithreaded, static, paste to a single line):
c
c     gfortran nufft3d_demof.f ../directft/dirft3df.f -o nufft3d_demof
c     ../../lib-static/libfinufftf.a -lstdc++ -lfftw3f -lfftw3f_omp -lm -fopenmp
c
      program nufft3d_demof
      implicit none
      
c     our fortran-header, always needed
      include 'finufft.fh'

      integer i,ier,iflag,j,k1,k2,k3,mx,n1,n2,n3
      integer*8 ms,mt,mu,nj,nk
      real*4, allocatable :: xj(:),yj(:),zj(:),sk(:),tk(:),uk(:)
      real*4 err,pi,eps,salg,ealg
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*8, allocatable :: cj(:),cj0(:),cj1(:),fk0(:),fk1(:)
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
               xj(j) = pi*cos(-pi*k1/n1)
               yj(j) = pi*cos(-pi*k2/n2)
               zj(j) = pi*cos(-pi*k3/n3)
               cj(j) = cmplx(sin(pi*j/n1),cos(pi*j/n2))
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
         if (i.eq.1) eps=1e-2
         if (i.eq.2) eps=1e-4
         if (i.eq.3) eps=1e-6
         if (i.eq.4) eps=1e-8
	 print*,' '
	 print*,' Requested precision eps =',eps
	 print*,' '
c
c     -----------------------
c     call 3D Type 1 method
c     -----------------------
c
         call dirft3d1f(nj,xj,yj,zj,cj,iflag,ms,mt,mu,fk0)
         call finufftf3d1(nj,xj,yj,zj,cj,iflag,
     1        eps,ms,mt,mu,fk1,defopts,ier)
         print *, ' ier = ',ier
         call errcomp(fk0,fk1,nk,err)
         print *, ' type 1 error = ',err
c
c     -----------------------
c      call 3D Type 2 method
c     -----------------------
         call dirft3d2f(nj,xj,yj,zj,cj0,iflag,ms,mt,mu,fk0)
         call finufftf3d2(nj,xj,yj,zj,cj1,iflag,eps,ms,mt,mu,fk0,
     1        defopts,ier)
         print *, ' ier = ',ier
         call errcomp(cj0,cj1,nj,err)
         print *, ' type 2 error = ',err
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
         call finufftf3d3(nj,xj,yj,zj,cj,iflag,eps,nk,sk,tk,uk,fk1,
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
      complex*8 fk0(n), fk1(n)
      real *4 salg,ealg,err
c
      ealg = 0e0
      salg = 0e0
      do k = 1, n
         ealg = ealg + cabs(fk1(k)-fk0(k))**2
         salg = salg + cabs(fk0(k))**2
      enddo
      err =sqrt(ealg/salg)
      return
      end
