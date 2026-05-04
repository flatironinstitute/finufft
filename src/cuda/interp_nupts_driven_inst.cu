// Per-dim instantiation TU: nupts-driven interp (gpu_method = 1).
// Compiled three times via CMake foreach with -DCUFINUFFT_DIM={1,2,3}.

#include "interp_nupts_driven.cuh"

#ifndef CUFINUFFT_DIM
#error "CUFINUFFT_DIM must be defined to 1, 2, or 3 (set by CMake)"
#endif

namespace cufinufft {
namespace spreadinterp {

template void do_interp_nupts_driven<float, CUFINUFFT_DIM>(
    const cufinufft_plan_t<float> &, cuda_complex<float> *, const cuda_complex<float> *,
    int);
template void do_interp_nupts_driven<double, CUFINUFFT_DIM>(
    const cufinufft_plan_t<double> &, cuda_complex<double> *,
    const cuda_complex<double> *, int);

} // namespace spreadinterp
} // namespace cufinufft

#if CUFINUFFT_DIM == 1
template void cufinufft_plan_t<float>::interp_nupts_driven(
    cuda_complex<float> *, const cuda_complex<float> *, int) const;
template void cufinufft_plan_t<double>::interp_nupts_driven(
    cuda_complex<double> *, const cuda_complex<double> *, int) const;
#endif
