#include "spread_impl.h"
#include "spread_dispatch.h"
#include "spread_poly_avx2_impl.h"
#include "spread_poly_scalar_impl.h"

namespace {

template <typename T> struct make_accumulator_tuple;

template <typename... Ts> struct make_accumulator_tuple<std::tuple<Ts...>> {
    using type = std::tuple<finufft::detail::VectorKernelAccumulator<Ts, Ts::out_width>...>;
};

using all_scalar_kernel_accumulators =
    typename make_accumulator_tuple<finufft::detail::all_scalar_kernels_tuple>::type;

// Currently, only selectively override scalar kernels when optimized versions are available.
using all_vector_float_kernel_accumulators = decltype(std::tuple_cat(
    std::declval<finufft::detail::all_avx2_float_accumulators_tuple>(),
    std::declval<all_scalar_kernel_accumulators>()));


template <typename VAcc, typename SAcc> struct MultiSubproblemFunctor {
    static constexpr int width = VAcc::width;
    static constexpr double beta = VAcc::beta;

    static_assert(VAcc::width == SAcc::width, "VAcc and SAcc must have the same width");
    static_assert(VAcc::beta == SAcc::beta, "VAcc and SAcc must have the same beta");

    template <typename T>
    void operator()(
        std::size_t offset, std::size_t size, T *__restrict du, std::size_t M,
        const T *__restrict kx, const T *__restrict dd) const {
        finufft::detail::spread_subproblem_1d_multi_impl(
            offset, size, du, M, kx, dd, width, VAcc{}, SAcc{});
    }
};

template <typename T> struct make_functors_tuple;
template <typename... Acc> struct make_functors_tuple<std::tuple<Acc...>> {
    using type = std::tuple<finufft::detail::SubproblemFunctor<Acc>...>;
};

using all_scalar_kernel_functors =
    typename make_functors_tuple<all_scalar_kernel_accumulators>::type;

using base_vector_float_kernel_functors =
    typename make_functors_tuple<all_vector_float_kernel_accumulators>::type;
using multi_vector_float_kernel_functors = std::tuple<
    MultiSubproblemFunctor<
        finufft::detail::ker_horner_avx2_w5_x3,
        finufft::detail::VectorKernelAccumulator<
            finufft::detail::ker_horner_scalar_3, finufft::detail::ker_horner_scalar_3::out_width>>,
    MultiSubproblemFunctor<
        finufft::detail::ker_horner_avx2_w4_x2,
        finufft::detail::VectorKernelAccumulator<
            finufft::detail::ker_horner_scalar_2,
            finufft::detail::ker_horner_scalar_2::out_width>>>;

using all_vector_float_kernel_functors = decltype(std::tuple_cat(
    std::declval<multi_vector_float_kernel_functors>(),
    std::declval<base_vector_float_kernel_functors>()));

} // namespace

namespace finufft {

namespace detail {

void spread_subproblem_1d_avx2(
    std::size_t off1, std::size_t size1, float *du, std::size_t M, const float *kx, const float *dd,
    int width, double es_beta, double es_c) noexcept {
    finufft::detail::DispatchSpecializedFromTuple<all_vector_float_kernel_functors> dispatch;
    dispatch(off1, size1, du, M, kx, dd, width, es_beta, es_c);
}

void spread_subproblem_1d_avx2(
    std::size_t off1, std::size_t size1, double *du, std::size_t M, const double *kx,
    const double *dd, int width, double es_beta, double es_c) noexcept {
    finufft::detail::DispatchSpecializedFromTuple<all_scalar_kernel_functors> dispatch;
    dispatch(off1, size1, du, M, kx, dd, width, es_beta, es_c);
}

} // namespace detail
} // namespace finufft