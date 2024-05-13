c     Using guru interface from fortran for 1D type 1 transforms,
c     a math test of one output, and how to change from default options.
c     Single-precision only.
c     Legacy-style: f77, plus dynamic allocation & derived types from f90.

c     To compile (linux/GCC) from this directory, use eg (paste to one line):
      
c     gfortran-9 -fopenmp -I../../include -I/usr/include guru1d1f.f
c     -L../../lib -lfinufftf -o guru1d1f

c     Alex Barnett and Libin Lu 5/29/20. Ptr fixes 10/6/21

      program guru1d1f
      implicit none

c     our fortran-header, always needed
      include 'finufft.fh'
c     if you want to use FFTW's modes by name...
      include 'fftw3.f'

c     note some inputs are int (int*4) but others BIGINT (int*8)
      integer ier,iflag
      integer*8 N,ktest,M,j,k,ktestindex,t1,t2,crate
      real*4, allocatable :: xj(:)
      real*4 err,tol,pi,t,fmax
      parameter (pi=3.141592653589793238462643383279502884197d0)
      complex*8, allocatable :: cj(:),fk(:)
      complex*8 fktest
      integer*8, allocatable :: n_modes(:)
      integer ttype,dim,ntrans
c     to pass null pointers to unused arguments...
      real*4, pointer :: dummy => null()

c     this is what you use as the "opaque" ptr to ptr to finufft_plan...
      integer*8 plan
c     this is how you create the options struct in fortran...
      type(finufft_opts) opts
c     or this is if you want default opts, make a null pointer...
      type(finufft_opts), pointer :: defopts => null()
   
c     how many nonuniform pts
      M = 200000
c     how many modes (not too much since FFTW_MEASURE slow later)
      N = 100000

      allocate(fk(N))
      allocate(xj(M))
      allocate(cj(M))
      print *,''
      print *,'creating data then run guru interface, default opts...'
c     create some quasi-random NU pts in [-pi, pi), complex strengths
      do j = 1,M
         xj(j) = pi * cos(pi*j/M)
         cj(j) = cmplx( sin((100e0*j)/M), cos(1.0+(50e0*j)/M))
      enddo

c     ---- SIMPLEST GURU DEMO WITH DEFAULT OPTS (as in simple1d1) ---------
c     mandatory parameters to FINUFFT guru interface... (ttype = trans type)
      ttype = 1
      dim = 1
      ntrans = 1
      iflag = 1
      tol = 1e-5
      allocate(n_modes(3))
      n_modes(1) = N
c     (note since dim=1, unused entries on n_modes are never read)
      call system_clock(t1)
c     use default options
      call finufftf_makeplan(ttype,dim,n_modes,iflag,ntrans,
     $     tol,plan,defopts,ier)
c     note for type 1 or 2, arguments 6-9 ignored...
      call finufftf_setpts(plan,M,xj,dummy,dummy,dummy,
     $     dummy,dummy,dummy,ier)
c     Do it: reads cj (strengths), writes fk (mode coeffs) and ier (status)
      call finufftf_execute(plan,cj,fk,ier)
      call system_clock(t2,crate)
      t = (t2-t1)/float(crate)
      if (ier.eq.0) then
         print '("done in ",f6.3," sec, ",e10.2" NU pts/s")',t,M/t
      else
         print *,'failed! ier=',ier
      endif
      call finufftf_destroy(plan,ier)

      
c     math test: single output mode with given freq (not array index) k
      ktest = N/3
      fktest = cmplx(0,0)
      do j=1,M
         fktest = fktest + cj(j) * cmplx( cos(ktest*xj(j)),
     $        sin(iflag*ktest*xj(j)) )
      enddo
c     compute inf norm of fk coeffs for use in rel err
      fmax = 0
      do k=1,N
         fmax = max(fmax,cabs(fk(k)))
      enddo
      ktestindex = ktest + N/2 + 1
      print '("rel err for mode k=",i10," is ",e10.2)',ktest,
     $     cabs(fk(ktestindex)-fktest)/fmax

c     ----------- GURU DEMO WITH NEW OPTIONS, MULTIPLE EXECS ----------
      print *,''
      print *, 'setting new options, rerun guru interface...'
      call finufftf_default_opts(opts)
c     refer to fftw3.f to set various FFTW plan modes...
      opts%fftw = FFTW_ESTIMATE_PATIENT
      opts%debug = 1
c     note you need a fresh plan if change opts
      call finufftf_makeplan(ttype,dim,n_modes,iflag,ntrans,
     $     tol,plan,opts,ier)
      call finufftf_setpts(plan,M,xj,dummy,dummy,dummy,
     $     dummy,dummy,dummy,ier)
c     Do it: reads cj (strengths), writes fk (mode coeffs) and ier (status)
      call finufftf_execute(plan,cj,fk,ier)
c     change the strengths
      do j = 1,M
         cj(j) = cmplx( sin((10e0*j)/M), cos(2.0+(20e0*j)/M))
      enddo
c     do another transform using same NU pts
      call finufftf_execute(plan,cj,fk,ier)
c     change the NU pts then do another transform w/ existing strengths...
      do j = 1,M
         xj(j) = pi/2.0 * cos(pi*j/M)
      enddo
      call finufftf_setpts(plan,M,xj,dummy,dummy,dummy,
     $     dummy,dummy,dummy,ier)
      call finufftf_execute(plan,cj,fk,ier)
      if (ier.eq.0) then
         print *,'done.'
      else
         print *,'failed! ier=',ier
      endif
      call finufftf_destroy(plan,ier)
      
      stop
      end
