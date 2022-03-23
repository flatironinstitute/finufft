#include <utility>

#include "spread_impl.h"

namespace {}

namespace finufft {

namespace detail {
void spread_subproblem_1d_scalar(
    std::size_t off1, std::size_t size1, float *du, std::size_t M, const float *kx, const float *dd,
    int width, double es_beta, double es_c) noexcept {
    auto accumulator = ScalarKernelAccumulator<float>{
        width, static_cast<float>(es_beta), static_cast<float>(es_c)};
    spread_subproblem_1d_impl(off1, size1, du, M, kx, dd, width, std::move(accumulator));
}

void spread_subproblem_1d_scalar(
    std::size_t off1, std::size_t size1, double *du, std::size_t M, const double *kx,
    const double *dd, int width, double es_beta, double es_c) noexcept {
    auto accumulator = ScalarKernelAccumulator<double>{width, es_beta, es_c};
    spread_subproblem_1d_impl(off1, size1, du, M, kx, dd, width, std::move(accumulator));
}

} // namespace detail

// Define dispatched version of the kernel
// This version automatically selects the desired instruction set
void spread_subproblem_1d(
    std::size_t off1, std::size_t size1, float *du, std::size_t M, const float *kx, const float *dd,
    int width, double es_beta, double es_c) noexcept {
    detail::spread_subproblem_1d_scalar(off1, size1, du, M, kx, dd, width, es_beta, es_c);
}
void spread_subproblem_1d(
    std::size_t off1, std::size_t size1, double *du, std::size_t M, const double *kx,
    const double *dd, int width, double es_beta, double es_c) noexcept {
    detail::spread_subproblem_1d_scalar(off1, size1, du, M, kx, dd, width, es_beta, es_c);
}

} // namespace finufft
