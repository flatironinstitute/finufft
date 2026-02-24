#include <finufft/detail/makeplan.hpp>

// Explicit instantiations for float and double.
// Private members are included because they are called from setpts() in
// finufft_core.cpp (a different translation unit).

template FINUFFT_PLAN_T<float>::FINUFFT_PLAN_T(int, int, const BIGINT *, int, int, float,
                                               const finufft_opts *, int &);
template FINUFFT_PLAN_T<double>::FINUFFT_PLAN_T(int, int, const BIGINT *, int, int, double,
                                                const finufft_opts *, int &);

template int FINUFFT_PLAN_T<float>::setup_spreadinterp();
template int FINUFFT_PLAN_T<double>::setup_spreadinterp();

template void FINUFFT_PLAN_T<float>::precompute_horner_coeffs();
template void FINUFFT_PLAN_T<double>::precompute_horner_coeffs();

template int FINUFFT_PLAN_T<float>::init_grid_kerFT_FFT();
template int FINUFFT_PLAN_T<double>::init_grid_kerFT_FFT();

template int finufft_makeplan_t<float>(int type, int dim, const BIGINT *n_modes, int iflag,
                                       int ntrans, float tol, FINUFFT_PLAN_T<float> **pp,
                                       const finufft_opts *opts);
template int finufft_makeplan_t<double>(int type, int dim, const BIGINT *n_modes,
                                        int iflag, int ntrans, double tol,
                                        FINUFFT_PLAN_T<double> **pp,
                                        const finufft_opts *opts);
