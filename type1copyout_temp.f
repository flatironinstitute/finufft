      integer ms,nf,klo(2),khi(2),jlo(2),jhi(2),err
      integer r,j,k

c     ...  fw and fwker is output of dft. fk is type-I output

      call indexmap1d(ns,mf,klo,khi,jlo,jhi,err)
c     copy the corrected DFT output into the fk output array
      do r=1,2
         j=jlo(r)
         do k=klo(r),khi(r)
            fk(k) = fw(j)/fwker(j)
            j=j+1
         enddo
      enddo






c     2D version
      integer ndims
      parameter (ndims = 2)
      integer ms,nf,klo(2,ndims),khi(2,ndims),jlo(2,ndims),jhi(2,ndims)
      integer err
      integer r1,r2,j1,k1,j2,k2

c     write into fwker1inv, fwker2inv

c     write into start to stop entries for each dim
      call indexmap1d(ns1,mf1,klo,khi,jlo,jhi,err)
      call indexmap1d(ns2,mf2,klo(1,2),khi(1,2),jlo(1,2),jhi(1,2),err)
c     copy the corrected DFT output into the fk output array
      do r2=1,2
         j2=jlo(r,2)
         do k2=klo(r,2),khi(r,2)
            do r1=1,2
               j1=jlo(r,1)

            fk(k) = fw(j)/fwker1(j)
            j=j+1
         enddo
      enddo

