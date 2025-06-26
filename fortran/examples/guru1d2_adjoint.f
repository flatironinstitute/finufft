c     Guru interface from fortran for adjoint of 1D type 2 transform,
c     (equiv to a type 2 with flipped iflag), math test of one output.
c     Double-precision only.
c     Legacy-style: f77, plus dynamic allocation & derived types from f90.

c     To compile (linux/GCC) from this directory, use eg (paste to one line):

c     gfortran -fopenmp -I../../include -I/usr/include guru1d2_adjoint.f
c     ../../lib/libfinufft.so -lfftw3 -lfftw3_omp -lgomp -lstdc++
c     -o guru1d2_adjoint

c     Martin Reinecke and Alex Barnett, 6/26/25

      program guru1d2_adjoint
      implicit none

c     our fortran header, always needed
      include 'finufft.fh'

c     note some inputs are int (int*4) but others BIGINT (int*8)
      integer ier,iflag
      integer*8 N,ktest,M,j,k,ktestindex,t1,t2,crate
      real*8, allocatable :: xj(:)
      real*8 err,tol,pi,t,fmax
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*16, allocatable :: cj(:),fk(:)
      complex*16 fktest
      integer*8, allocatable :: n_modes(:)
      integer ttype,dim,ntrans
c     to pass null pointers to unused arguments...
      real*8, pointer :: dummy => null()

c     this is what you use as the "opaque" ptr to ptr to finufft_plan...
      integer*8 plan
c     or this is if you want default opts, make a null pointer...
      type(finufft_opts), pointer :: defopts => null()


c     how many nonuniform pts
      M = 1000000
c     how many modes
      N = 100000

      allocate(fk(N))
      allocate(xj(M))
      allocate(cj(M))
      print *,''
      print *,'creating data then run guru interface, default opts...'
c     create some quasi-random NU pts in [-pi, pi), complex strengths
      do j = 1,M
         xj(j) = pi * dcos(pi*j/M)
         cj(j) = dcmplx( dsin((100d0*j)/M), dcos(1.0+(50d0*j)/M))
      enddo

c     mandatory parameters to FINUFFT guru interface...
      ttype = 2
      dim = 1
      ntrans = 1
      iflag = 1
      tol = 1d-9
      allocate(n_modes(3))
      n_modes(1) = N
c     (note since dim=1, unused entries on n_modes are never read)
      call system_clock(t1)
c     use default options
      call finufft_makeplan(ttype,dim,n_modes,iflag,ntrans,
     $     tol,plan,defopts,ier)
c     note for ttype 1 or 2, arguments 6-9 ignored...
      call finufft_setpts(plan,M,xj,dummy,dummy,dummy,
     $     dummy,dummy,dummy,ier)
c     Do the adjoint of planned transform:
c     reads fk (mode coeffs), writes cj (strengths) and ier (status)
      call finufft_execute_adjoint(plan,cj,fk,ier)
      call system_clock(t2,crate)
      t = (t2-t1)/float(crate)
      if (ier.eq.0) then
         print '("done in ",f6.3," sec, ",e10.2" NU pts/s")',t,M/t
      else
         print *,'failed! ier=',ier
      endif
      call finufft_destroy(plan,ier)


c     math test: single output mode with given freq (not array index) k
      ktest = N/3
      fktest = dcmplx(0,0)
      do j=1,M
c     note flipped iflag on the sin term, since adjoint...
         fktest = fktest + cj(j) * dcmplx( dcos(ktest*xj(j)),
     $        dsin(-iflag*ktest*xj(j)) )
      enddo
c     compute inf norm of fk coeffs for use in rel err
      fmax = 0
      do k=1,N
         fmax = max(fmax,cdabs(fk(k)))
      enddo
      ktestindex = ktest + N/2 + 1
      print '("rel err for mode k=",i10," is ",e10.2)',ktest,
     $     cdabs(fk(ktestindex)-fktest)/fmax

      stop
      end
