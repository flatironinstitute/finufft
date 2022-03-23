#include "spread_impl.h"
#include "spread_poly_scalar_impl.h"
#include "spread_poly_avx2_impl.h"

namespace {

template<typename T>
struct make_accumulator_tuple;

template<typename... Ts>
struct make_accumulator_tuple<std::tuple<Ts...>> {
    using type = std::tuple<finufft::detail::VectorKernelAccumulator<Ts, Ts::out_width>...>;
};

using all_scalar_kernel_accumulators = typename make_accumulator_tuple<finufft::detail::all_scalar_kernels_tuple>::type;

// Currently, only selectively override scalar kernels when optimized versions are available.
using all_vector_float_kernel_accumulators = decltype(std::tuple_cat(std::declval<std::tuple<finufft::detail::ker_horner_avx2_w7>>(), std::declval<all_scalar_kernel_accumulators>()));


template <typename... Accumulators> struct DispatchSpecialized;

template <typename Acc, typename... Accumulators> struct DispatchSpecialized<Acc, Accumulators...> {
    template <typename T>
    void operator()(
        std::size_t offset, std::size_t size, T *__restrict du, std::size_t M,
        const T *__restrict kx, const T *__restrict dd, int width, double es_beta,
        double es_c) const noexcept {
        if (Acc::width == width && std::abs(Acc::beta - es_beta) < 1e-8) {
            finufft::detail::spread_subproblem_1d_impl(offset, size, du, M, kx, dd, width, Acc{});
        } else {
            DispatchSpecialized<Accumulators...>{}(offset, size, du, M, kx, dd, width, es_beta, es_c);
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

void spread_subproblem_1d_avx2(
    std::size_t off1, std::size_t size1, float *du, std::size_t M, const float *kx, const float *dd,
    int width, double es_beta, double es_c) noexcept {
    DispatchSpecializedTuple<all_vector_float_kernel_accumulators> dispatch;
    dispatch(off1, size1, du, M, kx, dd, width, es_beta, es_c);
}

void spread_subproblem_1d_avx2(
    std::size_t off1, std::size_t size1, double *du, std::size_t M, const double *kx,
    const double *dd, int width, double es_beta, double es_c) noexcept {
    DispatchSpecializedTuple<all_scalar_kernel_accumulators> dispatch;
    dispatch(off1, size1, du, M, kx, dd, width, es_beta, es_c);
}

} // namespace detail
} // namespace finufft