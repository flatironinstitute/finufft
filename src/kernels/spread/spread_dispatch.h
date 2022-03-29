#pragma once

// Helpers for dispatching between various pre-compiled spread kernels

#include <utility>
#include "spread_impl.h"

namespace finufft {

namespace detail {

template <typename... Functors> struct DispatchSpecialized;

template <typename Fn, typename... Functors> struct DispatchSpecialized<Fn, Functors...> {
    template <typename T>
    void operator()(
        std::size_t offset, std::size_t size, T *__restrict du, std::size_t M,
        const T *__restrict kx, const T *__restrict dd, int width, double es_beta,
        double es_c) const noexcept {
        if (Fn::width == width && std::abs(Fn::beta - es_beta) < 1e-8) {
            Fn{}(offset, size, du, M, kx, dd);
        } else {
            DispatchSpecialized<Functors...>{}(offset, size, du, M, kx, dd, width, es_beta, es_c);
        }
    }
};

template <> struct DispatchSpecialized<> {
    template <typename T>
    void operator()(
        std::size_t offset, std::size_t size, T *du, std::size_t M, const T *kx, const T *dd,
        int width, double es_beta, double es_c) const noexcept {
        auto accumulator = finufft::detail::ScalarKernelAccumulator<T>{
            width, static_cast<T>(es_beta), static_cast<float>(es_c)};
        finufft::detail::spread_subproblem_1d_impl(
            offset, size, du, M, kx, dd, width, std::move(accumulator));
    }
};

template <typename T> struct DispatchSpecializedFromTuple;
template <typename... Ts>
struct DispatchSpecializedFromTuple<std::tuple<Ts...>> : finufft::detail::DispatchSpecialized<Ts...> {};


template <typename Acc> struct SubproblemFunctor {
    static constexpr int width = Acc::width;
    static constexpr double beta = Acc::beta;

    template <typename T>
    void operator()(
        std::size_t offset, std::size_t size, T *__restrict du, std::size_t M,
        const T *__restrict kx, const T *__restrict dd) const {
        finufft::detail::spread_subproblem_1d_impl(offset, size, du, M, kx, dd, width, Acc{});
    }
};


} // namespace detail

} // namespace finufft
