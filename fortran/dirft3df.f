cc Copyright (C) 2004-2009: Leslie Greengard and June-Yub Lee 
cc Contact: greengard@cims.nyu.edu
cc 
cc This software is being released under a FreeBSD license
cc (see license.txt in this directory). 
cc Single-prec version Barnett 4/5/17
cc
************************************************************************
      subroutine dirft3d1f(nj,xj,yj,zj,cj,iflag,ms,mt,mu,fk)
      implicit none
      integer nj, iflag, ms, mt, mu
      real*4 xj(nj), yj(nj), zj(nj)
      complex*8 cj(nj),fk(-ms/2:(ms-1)/2,-mt/2:(mt-1)/2,-mu/2:(mu-1)/2)
c ----------------------------------------------------------------------
c     direct computation of nonuniform FFT
c
c                      nj
c     fk(k1,k2,k3) =  SUM cj(j) exp(+/-i k1 xj(j)) *
c                     j=1       exp(+/-i k2 yj(j)) *
c                                 exp(+/-i k3 zj(j))
c
c     for -ms/2 <= k1 <= (ms-1)/2, 
c         -mt/2 <= k2 <= (mt-1)/2
c         -mu/2 <= k3 <= (mu-1)/2
c
c     If (iflag .ge.0) the + sign is used in the exponential.
c     If (iflag .lt.0) the - sign is used in the exponential.
c
************************************************************************
      integer j, k1, k2, k3
      complex*8 zf, cm1, cm2, z1n(-ms/2:(ms-1)/2), z2n(-mt/2:(mt-1)/2)
c
      do k3 = -mu/2, (mu-1)/2
         do k2 = -mt/2, (mt-1)/2
            do k1 = -ms/2, (ms-1)/2
               fk(k1,k2,k3) = cmplx(0d0,0d0)
            enddo
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
c     Precompute exponential for exp(+/-i k2 yj)
c     ----------------------------------------------------------
         if (iflag .ge. 0) then
            zf = cmplx(cos(yj(j)),+sin(yj(j)))
         else
            zf = cmplx(cos(yj(j)),-sin(yj(j)))
         endif
         z2n(0) = (1d0,0d0)
         do k1 = 1, (mt-1)/2
            z2n(k1) = zf*z2n(k1-1)
            z2n(-k1)= conjg(z2n(k1))
         enddo
         if (mt/2*2.eq.mt) z2n(-mt/2) = conjg(zf*z2n(mt/2-1))
c
c     ----------------------------------------------------------
c     Loop over k3 for zj
c     ----------------------------------------------------------
c
         if (iflag .ge. 0) then
            zf = cmplx(cos(zj(j)),+sin(zj(j)))
         else
            zf = cmplx(cos(zj(j)),-sin(zj(j)))
         endif
c
         cm2 = cj(j)
         do k3 = 0, (mu-1)/2
            do k2 = -mt/2, (mt-1)/2
               cm1 = cm2 * z2n(k2)
               do k1 = -ms/2, (ms-1)/2
                  fk(k1,k2,k3) = fk(k1,k2,k3) + cm1*z1n(k1)
               enddo
            enddo
            cm2 = zf*cm2
         enddo
c
         zf = conjg(zf)
         cm2 = cj(j)
         do k3 = -1, -mu/2, -1
            cm2 = zf*cm2
            do k2 = -mt/2, (mt-1)/2
               cm1 = cm2 * z2n(k2)
               do k1 = -ms/2, (ms-1)/2
                  fk(k1,k2,k3) = fk(k1,k2,k3) + cm1*z1n(k1)
               enddo
            enddo
         enddo

      enddo
      end
c
c
c
c
c
************************************************************************
      subroutine dirft3d2f(nj,xj,yj,zj,cj, iflag, ms,mt,mu,fk)
      implicit none
      integer nj, iflag, ms, mt, mu
      real*4 xj(nj), yj(nj), zj(nj)
      complex*8 cj(nj),fk(-ms/2:(ms-1)/2,-mt/2:(mt-1)/2,-mu/2:(mu-1)/2)
c ----------------------------------------------------------------------
c     direct computation of nonuniform FFT
c
c
c     cj(j) = SUM SUM SUM  fk(k1,k2,k3) exp(+/-i k1 xj(j)) * 
c             k1  k2  k3                exp(+/-i k2 yj(j))
c                                       exp(+/-i k3 zj(j))
c
c                            for j = 1,...,nj
c
c     where -ms/2 <= k1 <= (ms-1)/2 
c           -mt/2 <= k2 <= (mt-1)/2
c           -mu/2 <= k3 <= (mu-1)/2
c
c
c     If (iflag .ge.0) the + sign is used in the exponential.
c     If (iflag .lt.0) the - sign is used in the exponential.
c
************************************************************************
      integer j, k1, k2, k3
      complex*8 zf, cm1,cm2,cm3,z1n(-ms/2:(ms-1)/2),z2n(-mt/2:(mt-1)/2)
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
c
c     ----------------------------------------------------------
c     Precompute exponential for exp(+/-i k2 yj)
c     ----------------------------------------------------------
c
         if (iflag .ge. 0) then
            zf = cmplx(cos(yj(j)),+sin(yj(j)))
         else
            zf = cmplx(cos(yj(j)),-sin(yj(j)))
         endif
         z2n(0) = (1d0,0d0)
         do k2 = 1, (mt-1)/2
            z2n(k2) = zf*z2n(k2-1)
            z2n(-k2)= conjg(z2n(k2))
         enddo
         if (mt/2*2.eq.mt) z2n(-mt/2) = conjg(zf*z2n(mt/2-1))
c
c     ----------------------------------------------------------
c     Loop over k3 for zj
c     ----------------------------------------------------------
         if (iflag .ge. 0) then
            zf = cmplx(cos(zj(j)),+sin(zj(j)))
         else
            zf = cmplx(cos(zj(j)),-sin(zj(j)))
         endif
c
         cm2 = (0d0, 0d0)
         do k2 = -mt/2, (mt-1)/2
            cm1 = (0d0, 0d0)
            do k1 = -ms/2, (ms-1)/2
               cm1 = cm1 + z1n(k1) * fk(k1,k2,0)
            enddo
            cm2 = cm2 + z2n(k2) * cm1
         enddo
         cj(j) = cm2
c
         cm3 = zf
         do k3 = 1, (mu-1)/2
            cm2 = (0d0, 0d0)
            do k2 = -mt/2, (mt-1)/2
               cm1 = (0d0, 0d0)
               do k1 = -ms/2, (ms-1)/2
                 cm1 = cm1 + z1n(k1) * fk(k1,k2,k3)
               enddo
               cm2 = cm2 + z2n(k2) * cm1
            enddo
            cj(j) = cj(j) + cm3 * cm2

            cm2 = (0d0, 0d0)
            do k2 = -mt/2, (mt-1)/2
               cm1 = (0d0, 0d0)
               do k1 = -ms/2, (ms-1)/2
                 cm1 = cm1 + z1n(k1) * fk(k1,k2,-k3)
               enddo
               cm2 = cm2 + z2n(k2) * cm1
            enddo
            cj(j) = cj(j) + conjg(cm3) * cm2
            cm3 = cm3*zf
         enddo
c
         if (mu/2*2.eq.mu) then
            cm2 = (0d0, 0d0)
            do k2 = -mt/2, (mt-1)/2
               cm1 = (0d0, 0d0)
               do k1 = -ms/2, (ms-1)/2
                 cm1 = cm1 + z1n(k1) * fk(k1,k2,-mu/2)
               enddo
               cm2 = cm2 + z2n(k2) * cm1
            enddo
            cj(j) = cj(j) + conjg(cm3) * cm2
         endif
      enddo
      end
c
c
c
c
c
************************************************************************
      subroutine dirft3d3f(nj,xj,yj,zj,cj, iflag, nk,sk,tk,uk,fk)
      implicit none
      integer nj, iflag, nk
      real*4 xj(nj), yj(nj), zj(nj), sk(nk), tk(nk), uk(nk)
      complex*8 cj(nj), fk(nk)
c ----------------------------------------------------------------------
c     direct computation of nonuniform FFT
c
c              nj
c     fk(k) = SUM cj(j) exp(+/-i s(k) xj(j)) *
c             j=1       exp(+/-i t(k) yj(j)) *
c                       exp(+/-i u(k) zj(j))
c
c                    for k = 1, ..., nk
c
c     If (iflag .ge.0) the + sign is used in the exponential.
c     If (iflag .lt.0) the - sign is used in the exponential.
c
************************************************************************
      integer k, j
      real*4 ssk, stk, suk
c
      do k = 1, nk
         if (iflag .ge. 0) then
            ssk =  sk(k)
            stk =  tk(k)
            suk =  uk(k)
         else
            ssk =  -sk(k)
            stk =  -tk(k)
            suk =  -uk(k)
         endif
c
         fk(k) = cmplx(0d0,0d0)
         do j = 1, nj
            fk(k) = fk(k) + cj(j) * cmplx
     &        ( cos(ssk*xj(j)+stk*yj(j)+suk*zj(j)),
     &          sin(ssk*xj(j)+stk*yj(j)+suk*zj(j)) )
         enddo
      enddo
      end

