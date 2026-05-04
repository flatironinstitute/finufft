// Per-dim instantiation TU: output-driven spread (gpu_method = 3).
// Compiled three times via CMake foreach with -DCUFINUFFT_DIM={1,2,3}.
// Prep is shared with gpu_method=2 and instantiated in spread_subprob_inst.cu.

#include "spread_output_driven.cuh"

#ifndef CUFINUFFT_DIM
#error "CUFINUFFT_DIM must be defined to 1, 2, or 3 (set by CMake)"
#endif

namespace cufinufft {
namespace spreadinterp {

template void do_spread_output_driven<float, CUFINUFFT_DIM>(
    const cufinufft_plan_t<float> &, const cuda_complex<float> *, cuda_complex<float> *,
    int);
template void do_spread_output_driven<double, CUFINUFFT_DIM>(
    const cufinufft_plan_t<double> &, const cuda_complex<double> *,
    cuda_complex<double> *, int);

} // namespace spreadinterp
} // namespace cufinufft

#if CUFINUFFT_DIM == 1
template void cufinufft_plan_t<float>::spread_output_driven(
    const cuda_complex<float> *, cuda_complex<float> *, int) const;
template void cufinufft_plan_t<double>::spread_output_driven(
    const cuda_complex<double> *, cuda_complex<double> *, int) const;
#endif
