#include "onedim_nuft.h"
#include "dispatch.h"

#include <pmmintrin.h>
#include <xmmintrin.h>

namespace finufft {

DisableDenormals::DisableDenormals() noexcept
    : ftz_mode(_MM_GET_FLUSH_ZERO_MODE()), daz_mode(_MM_GET_DENORMALS_ZERO_MODE()) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

DisableDenormals::~DisableDenormals() noexcept {
    _MM_SET_FLUSH_ZERO_MODE(ftz_mode);
    _MM_SET_DENORMALS_ZERO_MODE(daz_mode);
}

// Utility macro to correctly invoke the kernel with the given suffix
#define FINUFFT_INVOKE_KERNEL(suffix, T)                                                           \
    [](size_t nk, size_t q, T const *f, T const *zf, T const *k, T *phihat) {                      \
        onedim_nuft_kernel_##suffix(nk, q, f, zf, k, phihat);                                      \
    }

void onedim_nuft_kernel(
    size_t nk, int q, float const *f, float const *z, float const *k, float *phihat) noexcept {
    static auto dispatch = make_dispatched_functor<void(
        size_t, int, float const *, float const *, float const *, float *)>(
        FINUFFT_INVOKE_KERNEL(scalar, float),
        FINUFFT_INVOKE_KERNEL(sse4, float),
        FINUFFT_INVOKE_KERNEL(avx2, float),
        FINUFFT_INVOKE_KERNEL(avx512, float));
    return dispatch(nk, q, f, z, k, phihat);
}

void onedim_nuft_kernel(
    size_t nk, int q, double const *f, double const *z, double const *k, double *phihat) noexcept {
    static auto dispatch = make_dispatched_functor<void(
        size_t, int, double const *, double const *, double const *, double *)>(
        FINUFFT_INVOKE_KERNEL(scalar, double),
        FINUFFT_INVOKE_KERNEL(sse4, double),
        FINUFFT_INVOKE_KERNEL(avx2, double),
        FINUFFT_INVOKE_KERNEL(avx512, double));
    return dispatch(nk, q, f, z, k, phihat);
}

#undef FINUFFT_INVOKE_KERNEL

} // namespace finufft
