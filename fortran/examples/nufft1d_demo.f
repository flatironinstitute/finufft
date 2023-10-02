c     Demo using FINUFFT for double-precision 1d transforms in legacy fortran.
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
c     Compile with, eg (GCC, multithreaded, static lib; paste to a single line):
c
c     gfortran nufft1d_demo.f ../directft/dirft1d.f -o nufft1d_demo
c     ../../lib-static/libfinufft.a -lstdc++ -lfftw3 -lfftw3_omp -lm -fopenmp
c
      program nufft1d_demo
      implicit none
      
c     our fortran-header, always needed
      include 'finufft.fh'
c
c --- local variables
c
      integer i,ier,iflag,j,k1,mx
      integer*8 ms,nj
      real*8, allocatable :: xj(:),sk(:)
      real*8 err,eps,pi
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*16, allocatable :: cj(:),cj0(:),cj1(:),fk0(:),fk1(:)
c     for default opts, make a null pointer...
      type(finufft_opts), pointer :: defopts => null()

c
c     --------------------------------------------------
c     create some test data
c     --------------------------------------------------
      ms = 90
      nj = 128
c     first alloc everything
      allocate(fk0(ms))
      allocate(fk1(ms))
      allocate(sk(ms))
      allocate(xj(nj))
      allocate(cj(nj))
      allocate(cj0(nj))
      allocate(cj1(nj))
      do k1 = -nj/2, (nj-1)/2
         j = k1+nj/2+1
         xj(j) = pi * dcos(-pi*j/nj)
         cj(j) = dcmplx( dsin(pi*j/nj), dcos(pi*j/nj))
      enddo
c
c     --------------------------------------------------
c     start tests
c     --------------------------------------------------
c
      iflag = 1
      print*,' Start 1D testing: ', ' nj =',nj, ' ms =',ms
      do i = 1,4
         if (i.eq.1) eps=1d-4
         if (i.eq.2) eps=1d-8
         if (i.eq.3) eps=1d-12
         if (i.eq.4) eps=1d-16
	 print*,' '
  	 print*,' Requested precision eps =',eps
	 print*,' '
c
c     -----------------------
c     call 1D Type1 method
c     -----------------------
c
         call dirft1d1(nj,xj,cj,iflag, ms,fk0)
         call finufft1d1(nj,xj,cj,iflag,eps,ms,fk1,defopts,ier)
         call errcomp(fk0,fk1,ms,err)
         print *,' ier = ',ier
         print *,' type 1 error = ',err
c
c     -----------------------
c     call 1D Type2 method
c     -----------------------
c
         call dirft1d2(nj,xj,cj0,iflag, ms,fk0,ier)
         call finufft1d2(nj,xj,cj1,iflag, eps, ms,fk0,defopts,ier)
         call errcomp(cj0,cj1,nj,err)
         print *,' ier = ',ier
         print *,' type 2 error = ',err
c
c     -----------------------
c     call 1D Type3 method
c     -----------------------
         do k1 = 1, ms
            sk(k1) = 48*dcos(k1*pi/ms)
         enddo
         call dirft1d3(nj,xj,cj,iflag, ms,sk,fk0)
         call finufft1d3(nj,xj,cj,iflag,eps, ms,sk,fk1,defopts,ier)
         call errcomp(cj0,cj1,nj,err)
         print *,' ier = ',ier
         print *,' type 3 error = ',err
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
