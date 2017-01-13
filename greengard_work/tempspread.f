      subroutine tempspread1d(nf1,fw,nj,xj,cj,itype,params)
      implicit none
      integer nf1,nj,n2,n3,nj3,i,itype
      real*8 xj(nj),pi
      real*8, allocatable :: xjscal(:) 
      real*8, allocatable :: yj(:) 
      real*8, allocatable :: zj(:) 
      real*8 params(4)
      complex*16 cj(nj)
      complex*16 fw(nf1)
c ----------------------------------------------------------------------

      pi = 4.0d0*datan(1.0d0)
      n2 = 1
      n3 = 1
      allocate(xjscal(nj))
      allocate(yj(nj))
      allocate(zj(nj))
c
      do i = 1,nj
         xjscal(i) = (xj(i)+pi)*nf1/(2*pi)
         yj(i) = 0.0d0
         zj(i) = 0.0d0
      enddo
      call cnufftspread_f(nf1,n2,n3,fw,nj,xjscal,yj,zj,cj,itype,params)
      return
      end
c
      subroutine tempspread2d(nf1,nf2,fw,nj,xj,yj,cj,itype,params)
      implicit none
      integer nf1,nf2,nj,n2,n3,nj3,i,itype
      real*8 xj(nj),yj(nj),pi
      real*8, allocatable :: xjscal(:) 
      real*8, allocatable :: yjscal(:) 
      real*8, allocatable :: zj(:) 
      real*8 params(4)
      complex*16 cj(nj)
      complex*16 fw(nf1,nf2)
c ----------------------------------------------------------------------

      pi = 4.0d0*datan(1.0d0)
      n3 = 1
      allocate(xjscal(nj))
      allocate(yjscal(nj))
      allocate(zj(nj))
c
      do i = 1,nj
         xjscal(i) = (xj(i)+pi)*nf1/(2*pi)
         yjscal(i) = (yj(i)+pi)*nf2/(2*pi)
         zj(i) = 0.0d0
      enddo
      call cnufftspread_f(nf1,nf2,n3,fw,nj,xjscal,yjscal,zj,
     1                    cj,itype,params)
      return
      end
c
      subroutine tempspread3d(nf1,nf2,nf3,fw,nj,xj,yj,zj,cj,
     1           itype,params)
      implicit none
      integer nf1,nf2,nf3,nj,n2,n3,nj3,i,itype
      real*8 xj(nj),yj(nj),zj(nj),pi
      real*8, allocatable :: xjscal(:) 
      real*8, allocatable :: yjscal(:) 
      real*8, allocatable :: zjscal(:) 
      real*8 params(4)
      complex*16 cj(nj)
      complex*16 fw(nf1,nf2,nf3)
c ----------------------------------------------------------------------

      pi = 4.0d0*datan(1.0d0)
      allocate(xjscal(nj))
      allocate(yjscal(nj))
      allocate(zjscal(nj))
c
      do i = 1,nj
         xjscal(i) = (xj(i)+pi)*nf1/(2*pi)
         yjscal(i) = (yj(i)+pi)*nf2/(2*pi)
         zjscal(i) = (zj(i)+pi)*nf3/(2*pi)
      enddo
      call cnufftspread_f(nf1,nf2,nf3,fw,nj,xjscal,yjscal,zjscal,
     1                    cj,itype,params)
      return
      end
