// Per-dim instantiation TU: nupts-driven spread (gpu_method = 1).
// Compiled three times by CMake (foreach dim), each invocation passing
// -DCUFINUFFT_DIM={1,2,3}; produces one object per dim.

#include "spread_nupts_driven.cuh"

#ifndef CUFINUFFT_DIM
#error "CUFINUFFT_DIM must be defined to 1, 2, or 3 (set by CMake)"
#endif

namespace cufinufft {
namespace spreadinterp {

template void do_spread_nupts_driven<float, CUFINUFFT_DIM>(
    const cufinufft_plan_t<float> &, const cuda_complex<float> *, cuda_complex<float> *,
    int);
template void do_spread_nupts_driven<double, CUFINUFFT_DIM>(
    const cufinufft_plan_t<double> &, const cuda_complex<double> *,
    cuda_complex<double> *, int);

template void do_prep_nupts_driven<float, CUFINUFFT_DIM>(cufinufft_plan_t<float> &);
template void do_prep_nupts_driven<double, CUFINUFFT_DIM>(cufinufft_plan_t<double> &);

} // namespace spreadinterp
} // namespace cufinufft

// Per-method plan-method instantiations live with the method body (not in
// spreadinterp.cpp, which is a pure-host TU and cannot pull in __global__
// kernels). Gate on dim=1 so each member is instantiated exactly once.
#if CUFINUFFT_DIM == 1
template void cufinufft_plan_t<float>::spread_nupts_driven(
    const cuda_complex<float> *, cuda_complex<float> *, int) const;
template void cufinufft_plan_t<double>::spread_nupts_driven(
    const cuda_complex<double> *, cuda_complex<double> *, int) const;
template void cufinufft_plan_t<float>::prep_nupts_driven();
template void cufinufft_plan_t<double>::prep_nupts_driven();
#endif
