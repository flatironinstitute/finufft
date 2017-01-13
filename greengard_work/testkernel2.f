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
      parameter (mx=10000)
      real*8 params(4)
      real*8 xj(mx),eps,pi,h,yj(mx),zj(mx),fj(mx)
      complex*16 cj(mx)
      real *8 sk(2*mx)
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
      n = 20
      h = 1.0d0/n
      do j = 1,n
         sk(j) = 1.0d0
      enddo
ccc      sk(10)=1.0d0
      ns = 5
      do j = 1,ns
ccc         xj(j) = -pi+ j*pi/ns
         xj(j) = 5+j
         yj(j) = 0.0d0
         zj(j) = 0.0d0
      enddo
      n2 = 1
      n3 = 1
      itype = 2
      call tempspread1d(n,sk,ns,xj,cj,itype,params) 
      call cnufftspread_f(n,n2,n3,sk,ns,xj,yj,zj,cj,itype,params)
      write(11,*) ' x = ['
      do j = 1,ns
         write(11,*) real(cj(j)), dimag(cj(j))
      enddo
      write(11,*) ' ];'
      stop
      end
