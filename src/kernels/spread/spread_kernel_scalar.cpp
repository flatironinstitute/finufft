#include "spread_impl.h"
#include "spread_poly_scalar_impl.h"

namespace {

template <typename... Kernels> struct DispatchSpecialized;

template <typename K, typename... Kernels> struct DispatchSpecialized<K, Kernels...> {
    template <typename T>
    void operator()(
        std::size_t offset, std::size_t size, T *__restrict du, std::size_t M,
        const T *__restrict kx, const T *__restrict dd, int width, double es_beta,
        double es_c) const noexcept {
        if (K::width == width && std::abs(K::beta - es_beta) < 1e-8) {
            finufft::detail::VectorKernelAccumulator<K, K::out_width> acc;
            finufft::detail::spread_subproblem_1d_impl(offset, size, du, M, kx, dd, width, acc);
        } else {
            DispatchSpecialized<Kernels...>{}(offset, size, du, M, kx, dd, width, es_beta, es_c);
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

template <typename T> struct DispatchSpecializedTuple;
template <typename... Ts>
struct DispatchSpecializedTuple<std::tuple<Ts...>> : DispatchSpecialized<Ts...> {};

} // namespace

namespace finufft {

namespace detail {

void spread_subproblem_1d_scalar(
    std::size_t off1, std::size_t size1, float *du, std::size_t M, const float *kx, const float *dd,
    int width, double es_beta, double es_c) noexcept {
    DispatchSpecializedTuple<all_scalar_kernels_tuple> dispatch;
    dispatch(off1, size1, du, M, kx, dd, width, es_beta, es_c);
}

void spread_subproblem_1d_scalar(
    std::size_t off1, std::size_t size1, double *du, std::size_t M, const double *kx,
    const double *dd, int width, double es_beta, double es_c) noexcept {
    DispatchSpecializedTuple<all_scalar_kernels_tuple> dispatch;
    dispatch(off1, size1, du, M, kx, dd, width, es_beta, es_c);
}

} // namespace detail
} // namespace finufft