#pragma once

/** Utilities for spreading
 *
 */

#include <cstddef>

namespace finufft {

#define FINUFFT_SPREAD_DECLARE_KERNEL_INSTRUCTION(EXT_NAME)                                        \
    void spread_subproblem_1d_##EXT_NAME(                                                          \
        std::size_t offset,                                                                        \
        std::size_t size,                                                                          \
        float *du,                                                                                 \
        std::size_t m,                                                                             \
        const float *kx,                                                                           \
        const float *dd,                                                                           \
        int width,                                                                                 \
        double es_beta,                                                                            \
        double es_c) noexcept;                                                                     \
    void spread_subproblem_1d_##EXT_NAME(                                                          \
        std::size_t offset,                                                                        \
        std::size_t size,                                                                          \
        double *du,                                                                                \
        std::size_t m,                                                                             \
        const double *kx,                                                                          \
        const double *dd,                                                                          \
        int width,                                                                                 \
        double es_beta,                                                                            \
        double es_c) noexcept;

namespace detail {
FINUFFT_SPREAD_DECLARE_KERNEL_INSTRUCTION(scalar)
FINUFFT_SPREAD_DECLARE_KERNEL_INSTRUCTION(sse4)
FINUFFT_SPREAD_DECLARE_KERNEL_INSTRUCTION(avx2)
FINUFFT_SPREAD_DECLARE_KERNEL_INSTRUCTION(avx512)
} // namespace detail

#undef FINUFFT_SPREAD_DECLARE_KERNEL_INSTRUCTION

// Define dispatched version of the kernel
// This version automatically selects the desired instruction set
void spread_subproblem_1d(
    std::size_t off1, std::size_t size1, float *du, std::size_t M, const float *kx, const float *dd,
    int width, double es_beta, double es_c) noexcept;
void spread_subproblem_1d(
    std::size_t off1, std::size_t size1, double *du, std::size_t M, const double *kx,
    const double *dd, int width, double es_beta, double es_c) noexcept;

}; // namespace finufft
