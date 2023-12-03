#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace finufft {

namespace detail {
/**! Baseline standard scalar implementation of the 1-d spread kernel.
 *
 * Note that this is very slow compared to the vectorized kernels.
 *
 */
template <typename T> struct ScalarKernelAccumulator {
    int width;
    T beta;
    T c;

    void operator()(T *output, T x1, T re, T im) const noexcept {
        for (int i = 0; i < width; ++i) {
            T xi = x1 + i;
            T k_val = std::exp(beta * std::sqrt(1 - c * xi * xi));
            output[2 * i] += re * k_val;
            output[2 * i + 1] += im * k_val;
        }
    }
};

/**! Adapter to transform a vector kernel evaluator into an accumulator
 * by separately evaluating into a temporary buffer then accumulating into the output.
 *
 */
template <typename K, int out_width> struct VectorKernelAccumulator {
    static constexpr int width = K::width;
    static constexpr double beta = K::beta;

    template <typename T> void operator()(T *output, T x1, T re, T im) const noexcept {
        K kernel;
        T ker[out_width];

        kernel(x1, ker);

        for (int i = 0; i < width; ++i) {
            output[2 * i] += re * ker[i];
            output[2 * i + 1] += im * ker[i];
        }
    }
};

template <typename T, typename Fn>
void spread_subproblem_1d_impl_mainloop(
    std::size_t size, T *__restrict du, std::size_t M, const T *__restrict kx,
    const T *__restrict dd, int width, Fn &&eval_and_accumulate) {

    T ns2 = static_cast<T>(width / 2.0);

    for (std::size_t i = 0; i < M; i++) { // loop over NU pts
        auto re0 = dd[2 * i];
        auto im0 = dd[2 * i + 1];

        // ceil offset, hence rounding, must match that in get_subgrid...
        std::size_t i1 = (std::size_t)std::ceil(kx[i] - ns2); // fine grid start index
        T x1 = (T)i1 - kx[i];                                 // x1 in [-w/2,-w/2+1], up to rounding
        // However if N1*epsmach>O(1) then can cause O(1) errors in x1, hence ppoly
        // kernel evaluation will fall outside their designed domains, >>1 errors.
        // This can only happen if the overall error would be O(1) anyway. Clip x1??

        // Clip x1
        if (x1 < -ns2)
            x1 = -ns2;
        if (x1 > -ns2 + 1)
            x1 = -ns2 + 1;

        eval_and_accumulate(du + 2 * i1, x1, re0, im0);
    }
}

// Helper to implement generic spreading kernels.
template <typename T, typename Fn>
void spread_subproblem_1d_impl(
    std::size_t offset, std::size_t size, T *__restrict du, std::size_t M, const T *__restrict kx,
    const T *__restrict dd, int width, Fn &&eval_and_accumulate) noexcept {

    T ns2 = static_cast<T>(width / 2.0); // half spread width

    std::fill_n(du, 2 * size, static_cast<T>(0.0));
    spread_subproblem_1d_impl_mainloop(
        size, du - 2 * offset, M, kx, dd, width, std::forward<Fn>(eval_and_accumulate));
}

template <typename T, typename FnV, typename Fn>
void spread_subproblem_1d_multi_impl(
    std::size_t offset, std::size_t size, T *__restrict du, std::size_t M, const T *__restrict kx,
    const T *__restrict dd, int width, FnV &&do_main, Fn &&eval_and_accumulate_scalar) noexcept {
    std::fill_n(du, 2 * size, static_cast<T>(0.0));
    std::size_t i = 0;

    // Main loop.
    for (; i < M - do_main.required_elements + 1; i += do_main.stride) {
        do_main(du - 2 * offset, kx, dd, i);
    }

    // Handle tail loop.
    spread_subproblem_1d_impl_mainloop(
        size,
        du - 2 * offset,
        M - i,
        kx + i,
        dd + 2 * i,
        width,
        std::forward<Fn>(eval_and_accumulate_scalar));
}

} // namespace detail
} // namespace finufft
