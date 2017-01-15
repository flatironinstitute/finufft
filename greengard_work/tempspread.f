c**********************************************************************
      subroutine tempspread1d(nf1,fw,nj,xj,cj,itype,params)
c**********************************************************************
c
c     wrapper for cnufftspread_f
c
c     INPUT:
c
c     nf1    number of points on oversampled uniform mesh  
c     nj     number of irregular points 
c     xj     location of irregular points on [-pi,pi].
c     itype  1 means irreg -> reg ***with wrapping***
c     itype  2 means reg -> irreg ***with wrapping***
c     params spreading kernel parameters from call to 
c            get_kernel_params_for_eps_f(params,eps)
c
c     IN/OUT:
c
c     fw     regular data (OUT if itype.eq.1, IN if itype.eq.2)
c     cj     irregular data (IN if itype.eq.1, OUT if itype.eq.2)
c ---------------------------------------------------------------------- 
c
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
c**********************************************************************
      subroutine tempspread2d(nf1,nf2,fw,nj,xj,yj,cj,itype,params)
c**********************************************************************
c
c     wrapper for cnufftspread_f
c
c     INPUT:
c
c     nf1    leading dimension of oversampled uniform mesh  
c     nf2    second dimension of oversampled uniform mesh  
c     nj     number of irregular points 
c     xj,yj  location of irregular points on [-pi,pi]^2.
c     itype  1 means irreg -> reg ***with wrapping***
c     itype  2 means reg -> irreg ***with wrapping***
c     params spreading kernel parameters from call to 
c            get_kernel_params_for_eps_f(params,eps)
c
c     IN/OUT:
c
c     fw     regular data (OUT if itype.eq.1, IN if itype.eq.2)
c     cj     irregular data (IN if itype.eq.1, OUT if itype.eq.2)
c ---------------------------------------------------------------------- 
c

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
c**********************************************************************
      subroutine tempspread3d(nf1,nf2,nf3,fw,nj,xj,yj,zj,cj,
     1           itype,params)
c**********************************************************************
c
c     wrapper for cnufftspread_f
c
c     INPUT:
c
c     nf1       leading dimension of oversampled uniform mesh  
c     nf2       second dimension of oversampled uniform mesh  
c     nf3       third dimension of oversampled uniform mesh  
c     nj        number of irregular points 
c     xj,yj,zj  location of irregular points on [-pi,pi]^2.
c     itype     1 means irreg -> reg ***with wrapping***
c     itype     2 means reg -> irreg ***with wrapping***
c     params    spreading kernel parameters from call to 
c               get_kernel_params_for_eps_f(params,eps)
c
c     IN/OUT:
c
c     fw     regular data (OUT if itype.eq.1, IN if itype.eq.2)
c     cj     irregular data (IN if itype.eq.1, OUT if itype.eq.2)
c ---------------------------------------------------------------------- 
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
