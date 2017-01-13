      subroutine dcfft3d(iflag,nf1,nf2,nf3,fw,fwsav1,fwsav2,fwsav3)
      implicit real *8 (a-h,o-z)
      complex *16 fw(nf1,nf2,nf3)
      complex *16 fwsav1(*)
      complex *16 fwsav2(*)
      complex *16 fwsav3(*)
      complex *16, allocatable :: ftemp2(:)
      complex *16, allocatable :: ftemp3(:)
c
c
c     ---------------------------------------------------------------
c     Compute 3D FFT 
c     ---------------------------------------------------------------
c
      allocate(ftemp2(nf2))
      allocate(ftemp3(nf3))
c
      do k2 = 1, nf2
      do k3 = 1, nf3
         if (iflag .ge. 0) then
            call dcfftb(nf1,fw(1,k2,k3),fwsav1)
         else
            call dcfftf(nf1,fw(1,k2,k3),fwsav1)
         endif
      enddo
      enddo
c
      do k1 = 1, nf1
      do k3 = 1, nf3
         do k2 = 1, nf2
            ftemp2(k2) = fw(k1,k2,k3)
         enddo
         if (iflag .ge. 0) then
           call dcfftb(nf2,ftemp2,fwsav2)
         else
           call dcfftf(nf2,ftemp2,fwsav2)
         endif
         do k2 = 1, nf2
            fw(k1,k2,k3) = ftemp2(k2) 
         enddo
      enddo
      enddo
c
c
      do k1 = 1, nf1
      do k2 = 1, nf2
         do k3 = 1, nf3
            ftemp3(k3) = fw(k1,k2,k3)
         enddo
         if (iflag .ge. 0) then
           call dcfftb(nf3,ftemp3,fwsav3)
         else
           call dcfftf(nf3,ftemp3,fwsav3)
         endif
         do k3 = 1, nf3
            fw(k1,k2,k3) = ftemp3(k3) 
         enddo
      enddo
      enddo
c
      return
      end
c
