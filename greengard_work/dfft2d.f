c***********************************************************************
      subroutine dcfft2d(iflag,nf1,nf2,fw,fwsav1,fwsav2)
c***********************************************************************
c     2D FFT code
c
c     INPUT:
c
c     iflag  determines which FFT is used ('forward' or 'backward')
c            .ge. zero -> dcfftb
c            .lt. zero -> dcfftf
c     nf1    leading dimension  of fw
c     nf2    second dimension  of fw
c     fw     complex 2D array
c     fwsav1 precomputed array needed by dcfft for dimension nf1  
c     fwsav2 precomputed array needed by dcfft for dimension nf2  
c
c     OUTPUT:
c
c     fw     overwritten by its transform
c ----------------------------------------------------------------------
c
      implicit real *8 (a-h,o-z)
      complex *16 fw(nf1,nf2)
      complex *16 fwsav1(*)
      complex *16 fwsav2(*)
      complex *16, allocatable :: ftemp(:)
c
c     ---------------------------------------------------------------
c     Compute 2D FFT 
c     ---------------------------------------------------------------
c
      allocate(ftemp(nf2))
      do k2 = 1, nf2
         if (iflag .ge. 0) then
            call dcfftb(nf1,fw(1,k2),fwsav1)
         else
            call dcfftf(nf1,fw(1,k2),fwsav1)
         endif
      enddo
c
      do k1 = 1, nf1
         do k2 = 1, nf2
            ftemp(k2) = fw(k1,k2)
         enddo
         if (iflag .ge. 0) then
           call dcfftb(nf2,ftemp,fwsav2)
         else
           call dcfftf(nf2,ftemp,fwsav2)
         endif
         do k2 = 1, nf2
            fw(k1,k2) = ftemp(k2) 
         enddo
      enddo
      return
      end
c
