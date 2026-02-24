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

template int FINUFFT_PLAN_T<FLT>::init_grid_kerFT_FFT();

template int finufft_makeplan_t<FLT>(int type, int dim, const BIGINT *n_modes, int iflag,
                                     int ntrans, FLT tol, FINUFFT_PLAN_T<FLT> **pp,
                                     const finufft_opts *opts);
