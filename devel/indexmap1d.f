      subroutine indexmap1d(ms,nf,klo,khi,jlo,jhi,ier)
c     Define start and end points of the two contiguous maps for 1d NUFFT
c     writing out of array at end of type-I or at start of type-II.
c     Makes the decision about even/odd regular-grid array lengths.
c
c     Inputs:
c     ms = regular grid length (index will be called k)
c     nf = upsampled DFT length (index called j). 
c     Outputs:
c     klo = list of 2 integers giving start indices for the two map segments
c           in the ms list.
c     khi = list of 2 integers giving end indices "
c     jlo,jhi = same as klo,khi but as indices in the upsampled DFT array
c     ier = 0 (success), else failure.
c
c     Note: length-ms list is defined by -ms/2 <= k <= (ms-1)/2.
c     length-nf list is defined by 0 <= j < nf.
c
c     Barnett 1/13/17
      implicit none
      integer ms,nf,klo(2),khi(2),jlo(2),jhi(2),ier

      ier = 0
      if (nf.le.ms) then
         write(*,*) 'indexmap1d problem: nf<ms!'
         ier = 1
         return
      endif
c     first segment of sum
      klo(1) = 0
c     rounds down if ms even...
      khi(1) = (ms-1)/2
      jlo(1) = klo(1)
      jhi(1) = khi(1)
c     -ms/2 if ms even, -(ms-1)/2 if odd...
      klo(2) = -(ms/2)
      khi(2) = -1
c     j indices are offset by nf
      jlo(2) = nf+klo(2)
      jhi(2) = nf+khi(2)
      end subroutine
