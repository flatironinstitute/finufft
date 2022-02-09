#include "onedim_nuft_impl.h"

namespace finufft {

INSTANTIATE_NUFT_IMPLEMENTATION_FOR_TYPE(onedim_nuft_kernel_sse4, 4, float);

// With SSE4, using vectorized instructions for double precision is slower than calling the scalar
// version.
void onedim_nuft_kernel_sse4(
    size_t nk, int q, double const *f, double const *z, double const *k, double *phihat) noexcept {
    onedim_nuft_kernel_scalar(nk, q, f, z, k, phihat);
}

} // namespace finufft
