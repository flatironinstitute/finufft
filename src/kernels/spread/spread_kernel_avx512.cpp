#include "spread_impl.h"
#include "spread_dispatch.h"
#include "spread_poly_avx512_impl.h"
#include "spread_poly_scalar_impl.h"

namespace finufft {
namespace detail {

void spread_subproblem_1d_avx512(
    std::size_t off1, std::size_t size1, float *du, std::size_t M, const float *kx, const float *dd,
    int width, double es_beta, double es_c) noexcept {
    finufft::detail::spread_subproblem_1d_multi_impl(
        off1, size1, du, M, kx, dd, width, ker_horner_avx512_w7_r4,
        finufft::detail::VectorKernelAccumulator<
            finufft::detail::ker_horner_scalar_5,
            finufft::detail::ker_horner_scalar_5::out_width>{});
}

void spread_subproblem_1d_avx512(
    std::size_t off1, std::size_t size1, double *du, std::size_t M, const double *kx,
    const double *dd, int width, double es_beta, double es_c) noexcept {
}

} // namespace detail

} // namespace finufft
