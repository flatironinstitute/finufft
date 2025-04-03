cc Copyright (C) 2004-2009: Leslie Greengard and June-Yub Lee
cc Contact: greengard@cims.nyu.edu
cc
cc This software is being release%d under a FreeBSD license
cc (see license.txt in this directory).
cc Single-prec version Barnett 4/5/17
cc
      subroutine dirft2d1f(nj,xj,yj,cj, iflag, ms,mt,fk)
      implicit none
      integer nj, iflag, ms, mt
      real*4 xj(nj), yj(nj)
      complex*8 cj(nj), fk(-ms/2:(ms-1)/2,-mt/2:(mt-1)/2)
c     ------------------------------------------------------------------
c     direct computation of nonuniform FFT
c
c                   nj
c     fk(k1,k2) =  SUM cj(j) exp(+/-i k1 xj(j)) exp(+/-i k2 yj(j))
c                  j=1
c
c     for -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2
c
c     If (iflag .ge.0) the + sign is used in the exponential.
c     If (iflag .lt.0) the - sign is used in the exponential.
c
c***********************************************************************
      integer j, k1, k2
      complex*8 zf, cm1, z1n(-ms/2:(ms-1)/2)
c
      do k2 = -mt/2, (mt-1)/2
         do k1 = -ms/2, (ms-1)/2
            fk(k1,k2) = cmplx(0d0,0d0)
         enddo
      enddo
c
      do j = 1, nj
c
c     ----------------------------------------------------------
c     Precompute exponential for exp(+/-i k1 xj)
c     ----------------------------------------------------------
c
         if (iflag .ge. 0) then
            zf = cmplx(cos(xj(j)),+sin(xj(j)))
         else
            zf = cmplx(cos(xj(j)),-sin(xj(j)))
         endif
         z1n(0) = (1d0,0d0)
         do k1 = 1, (ms-1)/2
            z1n(k1) = zf*z1n(k1-1)
            z1n(-k1)= conjg(z1n(k1))
         enddo
         if (ms/2*2.eq.ms) z1n(-ms/2) = conjg(zf*z1n(ms/2-1))
c
c     ----------------------------------------------------------
c     Loop over k2 for yj
c     ----------------------------------------------------------
         if (iflag .ge. 0) then
            zf = cmplx(cos(yj(j)),+sin(yj(j)))
         else
            zf = cmplx(cos(yj(j)),-sin(yj(j)))
         endif
c
         cm1 = cj(j)
         do k2 = 0, (mt-1)/2
            do k1 = -ms/2, (ms-1)/2
              fk(k1,k2) = fk(k1,k2) + cm1*z1n(k1)
            enddo
            cm1 = cm1*zf
         enddo
c
         zf = conjg(zf)
         cm1 = cj(j)
         do k2 = -1, -mt/2, -1
            cm1 = cm1*zf
            do k1 = -ms/2, (ms-1)/2
              fk(k1,k2) = fk(k1,k2) + cm1*z1n(k1)
            enddo
         enddo
      enddo
      end
c
c
c
c
c
c************************************************************************
      subroutine dirft2d2f(nj,xj,yj,cj, iflag, ms,mt,fk)
      implicit none
      integer nj, iflag, ms, mt
      real*4 xj(nj), yj(nj)
      complex*8 cj(nj), fk(-ms/2:(ms-1)/2,-mt/2:(mt-1)/2)
c     ----------------------------------------------------------------------
c     direct computation of nonuniform FFT
c
c
c     cj(j) = SUM  SUM  fk(k1,k2) exp(+/-i k1 xj(j)) exp(+/-i k2 yj(j))
c             k1   k2
c
c                            for j = 1,...,nj
c
c     where -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2
c
c
c     If (iflag .ge.0) the + sign is used in the exponential.
c     If (iflag .lt.0) the - sign is used in the exponential.
************************************************************************
      integer j, k1, k2
      complex*8 zf, cm1, cm2, z1n(-ms/2:(ms-1)/2)
c
      do j = 1, nj
c
c     ----------------------------------------------------------
c     Precompute exponential for exp(+/-i k1 xj)
c     ----------------------------------------------------------
         if (iflag .ge. 0) then
            zf = cmplx(cos(xj(j)),+sin(xj(j)))
         else
            zf = cmplx(cos(xj(j)),-sin(xj(j)))
         endif
         z1n(0) = (1d0,0d0)
         do k1 = 1, (ms-1)/2
            z1n(k1) = zf*z1n(k1-1)
            z1n(-k1)= conjg(z1n(k1))
         enddo
         if (ms/2*2.eq.ms) z1n(-ms/2) = conjg(zf*z1n(ms/2-1))
         if (iflag .ge. 0) then
            zf = cmplx(cos(yj(j)),+sin(yj(j)))
         else
            zf = cmplx(cos(yj(j)),-sin(yj(j)))
         endif
c
         cm1 = (0d0, 0d0)
         do k1 = -ms/2, (ms-1)/2
           cm1 = cm1 + z1n(k1) * fk(k1,0)
         enddo
         cj(j) = cm1
c
c     ----------------------------------------------------------
c     Loop over k2 for yj
c     ----------------------------------------------------------
c
         cm2 = zf
         do k2 = 1, (mt-1)/2
            cm1 = (0d0, 0d0)
            do k1 = -ms/2, (ms-1)/2
              cm1 = cm1 + z1n(k1) * fk(k1,k2)
            enddo
            cj(j) = cj(j) + cm2 * cm1

            cm1 = (0d0, 0d0)
            do k1 = -ms/2, (ms-1)/2
              cm1 = cm1 + z1n(k1) * fk(k1,-k2)
            enddo
            cj(j) = cj(j) + conjg(cm2) * cm1
            cm2 = cm2*zf
         enddo
c
         if (mt/2*2.eq.mt) then
            cm1 = (0d0, 0d0)
            do k1 = -ms/2, (ms-1)/2
              cm1 = cm1 + z1n(k1) * fk(k1,-mt/2)
            enddo
            cj(j) = cj(j) + conjg(cm2) * cm1
         endif

      enddo
      end

************************************************************************
      subroutine dirft2d3f(nj,xj,yj,cj, iflag, nk,sk,tk,fk)
      implicit none
      integer nj, iflag, nk
      real*4 xj(nj), yj(nj), sk(nk), tk(nk)
      complex*8 cj(nj), fk(nk)
c ----------------------------------------------------------------------
c     direct computation of nonuniform FFT
c
c              nj
c     fk(k) = SUM cj(j) exp(+/-i s(k) xj(j)) exp(+/-i t(k) yj(j))
c             j=1
c
c                    for k = 1, ..., nk
c
c     If (iflag .ge.0) the + sign is used in the exponential.
c     If (iflag .lt.0) the - sign is used in the exponential.
c
************************************************************************
      integer k, j
      real*4 ssk, stk
c
      do k = 1, nk
         if (iflag .ge. 0) then
            ssk =  sk(k)
            stk =  tk(k)
         else
            ssk =  -sk(k)
            stk =  -tk(k)
         endif
c
         fk(k) = cmplx(0d0,0d0)
         do j = 1, nj
            fk(k) = fk(k) + cj(j) * cmplx
     &        ( cos(ssk*xj(j)+stk*yj(j)), sin(ssk*xj(j)+stk*yj(j)) )
         enddo
      enddo
      end
