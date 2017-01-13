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
      ns = 1
      do j = 1,ns
         xj(j) = -20 + 40*j*h
         fj(j) = 1.0d0
         cj(j) = dcmplx(1.0d0,1.0d0)
         xj(j) = -pi
         yj(j) = 0.0d0
         zj(j) = 0.0d0
      enddo
      do j = 1,n
         write(6,*) xj(j)
      enddo
      if (2.eq.3) then 
         call evaluate_kernel_f(n,xj,sk,params) 
         write(11,*) ' x = ['
         do j = 1,n
            write(11,*) xj(j), sk(j)
         enddo
         write(11,*) ' ];'
         stop
      endif
      itype = 1
      n2 = 1
      n3 = 1
ccc      call cnufftspread_f(n,n2,n3,sk,ns,xj,yj,zj,cj,itype,params) 
      call tempspread1d(n,sk,ns,xj,cj,itype,params) 
      write(11,*) ' x = ['
      do j = 1,n
ccc         write(11,*) xj(j), sk(j)
         write(11,*) sk(2*j-1), sk(2*j)
      enddo
      write(11,*) ' ];'
      stop
      end
