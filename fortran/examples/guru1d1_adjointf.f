c     Guru interface from fortran for adjoint of 1D type 1 transform,
c     (a type 2 with flipped isign), a math test of one output.
c     Single-precision only.
c     Legacy-style: f77, plus dynamic allocation & derived types from f90.
c     To build and run it, see docs/fortran.rst.

c     Alex Barnett 10/1/26 based on guru1d1_adjoint.f

      program guru1d1_adjointf
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      implicit none

c     our fortran header, always needed
      include 'finufft.fh'

c     note some inputs are int (int*4) but others BIGINT (int*8)
      integer ier,iflag
      integer*8 N,jtest,M,j,k
      real*4, allocatable :: xj(:)
      real*4 err,tol,pi,cmax
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*8, allocatable :: cj(:),fk(:)
      complex*8 cjtest
      integer*8, allocatable :: n_modes(:)
      integer ttype,dim,ntrans
c     to pass null pointers to unused arguments...
      real*4, pointer :: dummy => null()

c     this is what you use as the "opaque" ptr to ptr to finufft_plan...
      integer*8 plan
c     or this is if you want default opts, make a null pointer...
      type(finufft_opts), pointer :: defopts => null()


c     how many nonuniform pts
      M = 1000000
c     how many modes (NB if too large lose acc in single prec)
      N = 10000

c     Note we use correct math indexing on the input mode array fk...
      allocate(fk(-N/2:(N-1)/2))
      allocate(xj(M))
      allocate(cj(M))
c     create some quasi-random NU pts in [-pi, pi), complex modes
      do j = 1,M
         xj(j) = pi * cos(pi*j/M)
      enddo
      do k = -N/2, (N-1)/2
         fk(k) = cmplx( sin(2.0+(100e0*k)/N), cos(1.0+(50e0*k)/N))
      enddo

c     We plan a type 1 (recalling its adjoint will be done)
      ttype = 1
      dim = 1
      ntrans = 1
      iflag = 1
      tol = 1e-3
      allocate(n_modes(3))
      n_modes(1) = N
c     (note since dim=1, unused entries on n_modes are never read)
c     use default options
      call finufftf_makeplan(ttype,dim,n_modes,iflag,ntrans,
     $     tol,plan,defopts,ier)
      if (ier.ne.0) then
         print *, 'FAILED: finufftf_makeplan ier is not 0'
         stop 1, quiet=.true.
      endif
c     note for ttype 1 or 2, arguments 6-9 ignored...
      call finufftf_setpts(plan,M,xj,dummy,dummy,dummy,
     $     dummy,dummy,dummy,ier)
      if (ier.ne.0) then
         print *, 'FAILED: finufftf_setpts ier is not 0'
         stop 1, quiet=.true.
      endif
c     Do adjoint of planned transform:
c     writes cj (strengths) and ier (status), reads fk (mode coeffs)
      call finufftf_execute_adjoint(plan,cj,fk,ier)
      if (ier.ne.0) then
         print *, 'FAILED: finufftf_execute_adjoint ier is not 0'
         stop 1, quiet=.true.
      endif
      call finufftf_destroy(plan,ier)


c     math test: single output target, note flipped isign
      jtest = 0.9*M
      cjtest = cmplx(0,0)
      do k = -N/2, (N-1)/2
         cjtest = cjtest + fk(k) * cmplx( cos(k*xj(jtest)),
     $        sin(-iflag*k*xj(jtest)) )
      enddo
c     compute inf norm of output vector for use in rel err
      cmax = 0
      do j=1,M
         cmax = max(cmax,cabs(cj(j)))
      enddo
c     max() skips NaN, so check the sum over all outputs
      err = cabs(cj(jtest)-cjtest)/cmax
c     (written so that a NaN err also fails)
      if (.not.ieee_is_finite(sum(abs(cj))) .or.
     $     .not.(err.le.10*tol)) then
         print *, 'FAILED: rel err too large, or NaN or Inf in output'
         stop 1, quiet=.true.
      endif
      print '("rel err = ",e10.2)',err
      end
