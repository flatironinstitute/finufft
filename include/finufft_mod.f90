module finufft_mod
! Fortran header recreating finufft_opts struct in fortran (f90 style)
! Module version, contributed by Reinhard Neder, 1/20/23.
! This must be kept synchronized with finufft_opts.h, matching its order.
! Also see ../fortran/finufftfort.cpp.
! Relies on "use ISO_C_BINDING" in the fortran module
use iso_c_binding
!
type finufft_opts
   integer(kind=C_INT) :: debug, spread_debug, spread_sort, spread_kerevalmeth
   integer(kind=C_INT) :: spread_kerpad, chkbnds, fftw, modeord
   real(kind=C_DOUBLE) :: upsampfac
   integer(kind=C_INT) :: spread_thread, maxbatchsize, showwarn, nthreads
   integer(kind=C_INT) :: spread_nthr_atomic, spread_max_sp_size
end type finufft_opts
!
end module finufft_mod
