// Per-dim instantiation TU: subproblem spread (gpu_method = 2).
// Compiled three times via CMake foreach with -DCUFINUFFT_DIM={1,2,3}.
// Also instantiates the prep helper shared with output-driven (gpu_method = 3).

#include "spread_subprob.cuh"

#ifndef CUFINUFFT_DIM
#error "CUFINUFFT_DIM must be defined to 1, 2, or 3 (set by CMake)"
#endif

namespace cufinufft {
namespace spreadinterp {

template void do_spread_subprob<float, CUFINUFFT_DIM>(const cufinufft_plan_t<float> &,
                                                      const cuda_complex<float> *,
                                                      cuda_complex<float> *, int);
template void do_spread_subprob<double, CUFINUFFT_DIM>(const cufinufft_plan_t<double> &,
                                                       const cuda_complex<double> *,
                                                       cuda_complex<double> *, int);

template void do_prep_subprob_and_OD<float, CUFINUFFT_DIM>(cufinufft_plan_t<float> &);
template void do_prep_subprob_and_OD<double, CUFINUFFT_DIM>(cufinufft_plan_t<double> &);

} // namespace spreadinterp
} // namespace cufinufft
