c
c     Notes: 
c     cnufftspread_f currently bulds 3d spreading array for each source.
c     Should be fixed to be dimensionally appropriate.
c
c
      program testkernel
      implicit none
c
      integer i,itype,ier,iflag,j,n,mx,ns,n2,n3
      parameter (mx=1000)
      real*8 params(4)
      real*8 xj(mx),eps,pi,h,yj(mx),zj(mx),fj(mx)
      complex*16 cj(mx)
      complex *16 sk(mx)
      complex *16 sk2(100,100)
      real*8 t0,t1,second
      parameter (pi=3.141592653589793238462643383279502884197d0)
c
c     --------------------------------------------------
c     create some test data
c     --------------------------------------------------
c
      open(unit=11,file='data.m')
      eps = 1.0d-7
      call get_kernel_params_for_eps_f(params,eps) 
      write(6,*) params(1)
      write(6,*) params(2)
      write(6,*) params(3)
      write(6,*) params(4)
      n = 100
      h = 1.0d0/n
      do j = 1,n
         sk(j) = 1.0d0
      enddo
      do i = 1,n
      do j = 1,n
         sk2(i,j) = 1.0d0
      enddo
      enddo
ccc      sk(10)=1.0d0
      ns = 1
      do j = 1,ns
         xj(j) = -pi+ j*pi/ns
         xj(1) = 0.0d0
         xj(2) = 0.2d0
         yj(1) = 0.0d0
         yj(2) = 0.1d0
         cj(j) = dcmplx(1.0d0,1.0d0)
ccc         xj(j) = 50+j
ccc         yj(j) = 0.0d0
         zj(j) = 0.0d0
      enddo
      n2 = 1
      n3 = 1
      itype = 2
      call tempspread1d(n,sk,ns,xj,cj,itype,params) 
ccc      call cnufftspread_f(n,n2,n3,sk,ns,xj,yj,zj,cj,itype,params)
      write(6,*) ' n = ',n
      write(11,*) ' x = ['
      do j = 1,ns
ccc         write(11,*) real(sk(j)), dimag(sk(j))
         write(11,*) real(cj(j)), dimag(cj(j))
      enddo
      write(11,*) ' ];'
c
      itype = 1
      call tempspread2d(n,n,sk2,ns,xj,yj,cj,itype,params) 
      do j = 1,n
      do i = 1,n
         write(12,*) real(sk2(i,j)), dimag(sk2(i,j))
      enddo
      enddo
      stop
      end
