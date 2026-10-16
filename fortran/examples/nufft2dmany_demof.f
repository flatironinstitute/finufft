c     Demo using FINUFFT for single-precision 2d transforms in legacy fortran.
c     Does types 1,2,3, including math test against direct summation.
c     Default opts only (see simple1d1f for how to change opts).
c     To build and run it, see docs/fortran.rst.
c
c     A modification of drivers from the CMCL NUFFT, (C) 2004-2009,
c     Leslie Greengard and June-Yub Lee. See: cmcl_license.txt.
c
c     Vectorized (many data vectors) demo type 1,2 by Melody Shih, 2018,
c     type 3 & single-prec by Alex Barnett, 2020. Based on nufft2d_demo.f.
c     Also see: ../README.
c
      program nufft2dmany_demof
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      implicit none

c     our fortran-header, always needed
      include 'finufft.fh'
c
      integer i,ier,iflag,j,k1,k2,mx,n1,n2,ntrans,d
      integer*8 ms,mt,nj,nk
      real*4, allocatable :: xj(:),yj(:),sk(:),tk(:)
      real*4 err,pi,tol,maxerr
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*8, allocatable :: cj(:),cj0(:),cj1(:),fk0(:),fk1(:)
c     for default opts, make a null pointer...
      type(finufft_opts), pointer :: defopts => null()
c
c     --------------------------------------------------
c     create some test data
c     --------------------------------------------------
c     Here ntrans will be the number of vectors of strength data for the
c     same set of nonuniform points:
      ntrans = 2
c     As with nufft2d_demof.f, nj is "M" the # NU pts, and nk is "N", # modes:
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
      allocate(sk(nk))
      allocate(tk(nk))
      allocate(fk0(nk*ntrans))
      allocate(fk1(nk*ntrans))
      allocate(cj (nj*ntrans))
      allocate(cj0(nj*ntrans))
      allocate(cj1(nj*ntrans))
      do k1 = -n1/2, (n1-1)/2
         do k2 = -n2/2, (n2-1)/2
            j = (k2+n2/2+1) + (k1+n1/2)*n2
            xj(j) = pi*cos(-pi*k1/n1)
            yj(j) = pi*cos(-pi*k2/n2)
            do d = 0, ntrans-1
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
      do i = 1,3
         if (i.eq.1) tol=1e-2
         if (i.eq.2) tol=1e-4
         if (i.eq.3) tol=1e-5
c
c     -----------------------
c     call 2D Type 1 method
c     -----------------------
c
         call finufftf2d1many(ntrans,nj,xj,yj,cj,iflag,
     &                         tol,ms,mt,fk1,defopts,ier)
         if (ier.ne.0) then
            print *, 'FAILED: finufftf2d1many ier is not 0'
            stop 1, quiet=.true.
         endif
         do d = 1, ntrans
            call dirft2d1f(nj,xj,yj,cj(1+(d-1)*nj:d*nj),iflag,ms,mt,
     &                    fk0(1+(d-1)*nk:d*nk))
            call errcomp(fk0(1+(d-1)*nk:d*nk),fk1(1+(d-1)*nk:d*nk),
     &                   nk,err)
            if (.not.ieee_is_finite(sum(abs(fk1(1+(d-1)*nk:d*nk))))
     $           .or. .not.(err.le.10*tol)) then
            print *, 'FAILED: type 1 rel err too large, or NaN or Inf'
               stop 1, quiet=.true.
            endif
            maxerr = max(maxerr,err)
         enddo
c
c     -----------------------
c      call 2D Type 2 method
c     -----------------------
         call finufftf2d2many(ntrans,nj,xj,yj,cj1,iflag,
     &                         tol,ms,mt,fk0,defopts,ier)
         if (ier.ne.0) then
            print *, 'FAILED: finufftf2d2many ier is not 0'
            stop 1, quiet=.true.
         endif
         do d = 1, ntrans
            call dirft2d2f(nj,xj,yj,cj0(1+(d-1)*nj:d*nj),iflag,ms,mt,
     &                    fk0(1+(d-1)*nk:d*nk))
            call errcomp(cj0(1+(d-1)*nj:d*nj),cj1(1+(d-1)*nj:d*nj),
     &                   nj,err)
            if (.not.ieee_is_finite(sum(abs(cj1(1+(d-1)*nj:d*nj))))
     $           .or. .not.(err.le.10*tol)) then
            print *, 'FAILED: type 2 rel err too large, or NaN or Inf'
               stop 1, quiet=.true.
            endif
            maxerr = max(maxerr,err)
         enddo
c
c     -----------------------
c      call 2D Type3 method
c     -----------------------
         do k1 = 1, nk
            sk(k1) = 48*(cos(k1*pi/nk))
            tk(k1) = 32*(sin(-pi/2+k1*pi/nk))
         enddo

         call finufftf2d3many(ntrans,nj,xj,yj,cj,iflag,tol,nk,sk,tk,
     &        fk1,defopts,ier)
         if (ier.ne.0) then
            print *, 'FAILED: finufftf2d3many ier is not 0'
            stop 1, quiet=.true.
         endif
         do d = 1, ntrans
            call dirft2d3f(nj,xj,yj,cj(1+(d-1)*nj:d*nj),iflag,nk,
     &           sk,tk,fk0(1+(d-1)*nk:d*nk))
            call errcomp(fk0(1+(d-1)*nk:d*nk),fk1(1+(d-1)*nk:d*nk),
     &                   nk,err)
            if (.not.ieee_is_finite(sum(abs(fk1(1+(d-1)*nk:d*nk))))
     $           .or. .not.(err.le.10*tol)) then
            print *, 'FAILED: type 3 rel err too large, or NaN or Inf'
               stop 1, quiet=.true.
            endif
            maxerr = max(maxerr,err)
         enddo
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
      complex*8 fk0(n), fk1(n)
      real *4 fmax,emax,err
c
      emax = 0e0
      fmax = 0e0
      do k = 1, n
         emax = max(emax,cabs(fk1(k)-fk0(k)))
         fmax = max(fmax,cabs(fk1(k)))
      enddo
      err = emax/fmax
      return
      end
