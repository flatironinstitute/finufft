module finufft_mod
! Fortran header recreating finufft_opts struct in fortran (f90 style)
! Module version, contributed by Reinhard Neder, 1/20/23. Order fixed 1/7/25.
! This must be kept synchronized with finufft_opts.h, matching its order.
! Also see ../fortran/finufftfort.cpp.
! Relies on "use ISO_C_BINDING" in the fortran module.
use iso_c_binding
type finufft_opts

   ! data handling opts...
   integer(kind=C_INT) :: modeord, spreadinterponly

   ! diagnostic opts...
   integer(kind=C_INT) :: debug, spread_debug, showwarn

   ! alg perf opts...
   integer(kind=C_INT) :: nthreads,fftw,spread_sort,spread_kerevalmeth
   integer(kind=C_INT) :: spread_kerpad
   real(kind=C_DOUBLE) :: upsampfac
   integer(kind=C_INT) :: spread_thread, maxbatchsize
   integer(kind=C_INT) :: spread_nthr_atomic, spread_max_sp_size
   integer(kind=C_SIZE_T) :: fftw_lock_fun, fftw_unlock_fun, fftw_lock_data
   !  really, last should be type(C_PTR) :: etc, but fails to print nicely

end type finufft_opts
end module finufft_mod
