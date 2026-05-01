// Per-dim instantiation TU: subproblem interp (gpu_method = 2).
// Compiled three times via CMake foreach with -DCUFINUFFT_DIM={1,2,3}.

#include "interp_subprob.cuh"

#ifndef CUFINUFFT_DIM
#error "CUFINUFFT_DIM must be defined to 1, 2, or 3 (set by CMake)"
#endif

namespace cufinufft {
namespace spreadinterp {

template void do_interp_subprob<float, CUFINUFFT_DIM>(const cufinufft_plan_t<float> &,
                                                      cuda_complex<float> *,
                                                      const cuda_complex<float> *, int);
template void do_interp_subprob<double, CUFINUFFT_DIM>(const cufinufft_plan_t<double> &,
                                                       cuda_complex<double> *,
                                                       const cuda_complex<double> *, int);

} // namespace spreadinterp
} // namespace cufinufft
