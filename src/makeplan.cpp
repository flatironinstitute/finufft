#include <finufft/detail/makeplan.hpp>

// Explicit instantiations, selected by FINUFFT_SINGLE define.
// Private members are included because they are called from setpts
// (a different translation unit).

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

template FINUFFT_PLAN_T<FLT>::FINUFFT_PLAN_T(int, int, const BIGINT *, int, int, FLT,
                                             const finufft_opts *, int &);

template int FINUFFT_PLAN_T<FLT>::setup_spreadinterp();

template void FINUFFT_PLAN_T<FLT>::precompute_horner_coeffs();

// These are called by init_grid_kerFT_FFT (instantiated in fft.cpp), so must be
// explicitly instantiated here where their bodies (in detail/makeplan.hpp) are visible.
template int FINUFFT_PLAN_T<FLT>::set_nf_type12(BIGINT ms, BIGINT *nf) const;
template void FINUFFT_PLAN_T<FLT>::onedim_fseries_kernel(BIGINT nf,
                                                          std::vector<FLT> &fwkerhalf) const;

extern template void FINUFFT_PLAN_T<FLT>::create_fft_plan();    // instantiated in fft.cpp
extern template int FINUFFT_PLAN_T<FLT>::init_grid_kerFT_FFT(); // instantiated in fft.cpp

template int finufft_makeplan_t<FLT>(int type, int dim, const BIGINT *n_modes, int iflag,
                                     int ntrans, FLT tol, FINUFFT_PLAN_T<FLT> **pp,
                                     const finufft_opts *opts);
