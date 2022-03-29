#include "spread_impl.h"
#include "spread_dispatch.h"
#include "spread_poly_scalar_impl.h"

namespace {

template <typename T> struct make_functor_tuple;

template <typename... Ts> struct make_functor_tuple<std::tuple<Ts...>> {
    using type = std::tuple<finufft::detail::SubproblemFunctor<finufft::detail::VectorKernelAccumulator<Ts, Ts::out_width>>...>;
};

using all_scalar_functors =
    typename make_functor_tuple<finufft::detail::all_scalar_kernels_tuple>::type;

} // namespace

namespace finufft {

namespace detail {

void spread_subproblem_1d_scalar(
    std::size_t off1, std::size_t size1, float *du, std::size_t M, const float *kx, const float *dd,
    int width, double es_beta, double es_c) noexcept {
    finufft::detail::DispatchSpecializedFromTuple<all_scalar_functors> dispatch;
    dispatch(off1, size1, du, M, kx, dd, width, es_beta, es_c);
}

void spread_subproblem_1d_scalar(
    std::size_t off1, std::size_t size1, double *du, std::size_t M, const double *kx,
    const double *dd, int width, double es_beta, double es_c) noexcept {
    finufft::detail::DispatchSpecializedFromTuple<all_scalar_functors> dispatch;
    dispatch(off1, size1, du, M, kx, dd, width, es_beta, es_c);
}

} // namespace detail
} // namespace finufft