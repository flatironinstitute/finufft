#include <utility>

#include "spread.h"

namespace finufft {

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
