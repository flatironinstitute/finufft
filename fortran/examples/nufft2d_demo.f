c     Demo using FINUFFT for double-precision 2d transforms in legacy fortran.
c     Does types 1,2,3, including math test against direct summation.
c     Default opts only (see simple1d1 for how to change opts).
c     To build and run it, see docs/fortran.rst.
c
c     A slight modification of drivers from the CMCL NUFFT, (C) 2004-2009,
c     Leslie Greengard and June-Yub Lee. See: cmcl_license.txt.
c
c     Tweaked by Alex Barnett to call FINUFFT 2/17/17.
c     dyn malloc; type 2 uses same input data fk0, 3/8/17
c     Also see: ../README.
c
      program nufft2d_demo
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      implicit none

c     our fortran-header, always needed
      include 'finufft.fh'
c
      integer i,ier,iflag,j,k1,k2,mx,n1,n2
      integer*8 nj,ms,mt,nk
      real*8, allocatable :: xj(:),yj(:),sk(:),tk(:)
      real*8 err,maxerr,pi,tol
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*16, allocatable :: cj(:),cj0(:),cj1(:),fk0(:),fk1(:)
c     for default opts, make a null pointer...
      type(finufft_opts), pointer :: defopts => null()

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
            xj(j) = pi*dcos(-pi*k1/n1)
            yj(j) = pi*dcos(-pi*k2/n2)
            cj(j) = dcmplx(dsin(pi*j/n1),dcos(pi*j/n2))
         enddo
      enddo
c
c     -----------------------
c     start tests
c     -----------------------
c
      iflag = 1
      maxerr = 0d0
      do i = 1,4
         if (i.eq.1) tol=1d-3
         if (i.eq.2) tol=1d-6
         if (i.eq.3) tol=1d-9
         if (i.eq.4) tol=1d-12
c
c     -----------------------
c     call 2D Type 1 method
c     -----------------------
c
         call dirft2d1(nj,xj,yj,cj,iflag,ms,mt,fk0)
         call finufft2d1(nj,xj,yj,cj,iflag,tol,ms,mt,fk1,defopts,ier)
         if (ier.ne.0) then
            print *, 'FAILED: finufft2d1 ier is not 0'
            stop 1, quiet=.true.
         endif
         call errcomp(fk0,fk1,nk,err)
         if (.not.ieee_is_finite(sum(abs(fk1))) .or.
     $        .not.(err.le.10*tol)) then
            print *, 'FAILED: type 1 rel err too large, or NaN or Inf'
            stop 1, quiet=.true.
         endif
         maxerr = max(maxerr,err)
c
c     -----------------------
c      call 2D Type 2 method
c     -----------------------
         call dirft2d2(nj,xj,yj,cj0,iflag,ms,mt,fk0)
         call finufft2d2(nj,xj,yj,cj1,iflag,tol,ms,mt,fk0,defopts,ier)
         if (ier.ne.0) then
            print *, 'FAILED: finufft2d2 ier is not 0'
            stop 1, quiet=.true.
         endif
         call errcomp(cj0,cj1,nj,err)
         if (.not.ieee_is_finite(sum(abs(cj1))) .or.
     $        .not.(err.le.10*tol)) then
            print *, 'FAILED: type 2 rel err too large, or NaN or Inf'
            stop 1, quiet=.true.
         endif
         maxerr = max(maxerr,err)
c
c     -----------------------
c      call 2D Type3 method
c     -----------------------
         do k1 = 1, nk
            sk(k1) = 48*(dcos(k1*pi/nk))
            tk(k1) = 32*(dsin(-pi/2+k1*pi/nk))
         enddo

         call dirft2d3(nj,xj,yj,cj,iflag,nk,sk,tk,fk0)
         call finufft2d3(nj,xj,yj,cj,iflag,tol,nk,sk,tk,fk1,defopts,ier)
         if (ier.ne.0) then
            print *, 'FAILED: finufft2d3 ier is not 0'
            stop 1, quiet=.true.
         endif
         call errcomp(fk0,fk1,nk,err)
         if (.not.ieee_is_finite(sum(abs(fk1))) .or.
     $        .not.(err.le.10*tol)) then
            print *, 'FAILED: type 3 rel err too large, or NaN or Inf'
            stop 1, quiet=.true.
         endif
         maxerr = max(maxerr,err)
      enddo
      print '("max rel err = ",e10.2)',maxerr
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
      real *8 fmax,emax,err
c
      emax = 0d0
      fmax = 0d0
      do k = 1, n
         emax = max(emax,cdabs(fk1(k)-fk0(k)))
         fmax = max(fmax,cdabs(fk1(k)))
      enddo
      err = emax/fmax
      return
      end
