// Per-dim instantiation TU: 3D block-gather spread (gpu_method = 4), Ndim = 3.

#include "spread_blockgather.cuh"

namespace cufinufft {
namespace spreadinterp {

template void do_spread_blockgather_3d<float>(const cufinufft_plan_t<float> &,
                                              const cuda_complex<float> *,
                                              cuda_complex<float> *, int);
template void do_spread_blockgather_3d<double>(const cufinufft_plan_t<double> &,
                                               const cuda_complex<double> *,
                                               cuda_complex<double> *, int);

template void do_indexSort_blockgather_3d<float>(cufinufft_plan_t<float> &);
template void do_indexSort_blockgather_3d<double>(cufinufft_plan_t<double> &);

} // namespace spreadinterp
} // namespace cufinufft

template void cufinufft_plan_t<float>::spread_blockgather_3d(
    const cuda_complex<float> *, cuda_complex<float> *, int) const;
template void cufinufft_plan_t<double>::spread_blockgather_3d(
    const cuda_complex<double> *, cuda_complex<double> *, int) const;
template void cufinufft_plan_t<float>::indexSort_blockgather_3d();
template void cufinufft_plan_t<double>::indexSort_blockgather_3d();
