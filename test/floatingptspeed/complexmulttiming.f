c     gfortran complexmulttiming.f -o complexmulttimingf -O3
c     ./complexmulttimingf
      program complexmulttiming
      implicit none
      integer m,i
      real *8, allocatable :: x(:),x2(:)
      complex*16, allocatable :: z(:),z2(:)
      complex*16 ima
      real :: t0,t1
      data ima/(0.0d0,1.0d0)/
      m=1e8

c     real
      allocate(x(m))
      allocate(x2(m))
      do i=1,m
         x(i) = rand()
         x2(i) = rand()
      enddo
      call cpu_time(t0)
      do i=1,m
         x(i) = x(i) * x2(i)
      enddo
      call cpu_time(t1)
      write (*,'(I10," fortran real*8 mults in ",f6.3," s")'), m, t1-t0
      deallocate(x)
      deallocate(x2)

c     complex
      allocate(z(m))
      allocate(z2(m))
      do i=1,m
         z(i) = rand() + ima*rand()
         z2(i) = rand() + ima*rand()
      enddo
      call cpu_time(t0)
      do i=1,m
         z(i) = z(i) * z2(i)
      enddo
      call cpu_time(t1)
      write (*,'(I10," fortran complex*16 mults in ",f6.3," s")'), m,
     c     t1-t0

      end program
