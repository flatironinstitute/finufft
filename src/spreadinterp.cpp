#include <finufft/spreadinterp.hpp>

// Per-dimension functions are explicitly instantiated in spreadinterp_1d/2d/3d.cpp.
// This TU handles the remaining per-precision symbols.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

// Suppress re-instantiation of the per-dim symbols defined elsewhere:
extern template void FINUFFT_PLAN_T<FLT>::spread_subproblem_1d(
    const Subgrid &, FLT *, UBIGINT, const FLT *, const FLT *, const FLT *,
    const FLT *) const noexcept;
extern template void FINUFFT_PLAN_T<FLT>::interp_subproblem_1d(
    const Subgrid &, const FLT *, UBIGINT, const FLT *, const FLT *, const FLT *,
    const BIGINT *, FLT *) const noexcept;
extern template void FINUFFT_PLAN_T<FLT>::spread_subproblem_2d(
    const Subgrid &, FLT *, UBIGINT, const FLT *, const FLT *, const FLT *,
    const FLT *) const noexcept;
extern template void FINUFFT_PLAN_T<FLT>::interp_subproblem_2d(
    const Subgrid &, const FLT *, UBIGINT, const FLT *, const FLT *, const FLT *,
    const BIGINT *, FLT *) const noexcept;
extern template void FINUFFT_PLAN_T<FLT>::spread_subproblem_3d(
    const Subgrid &, FLT *, UBIGINT, const FLT *, const FLT *, const FLT *,
    const FLT *) const noexcept;
extern template void FINUFFT_PLAN_T<FLT>::interp_subproblem_3d(
    const Subgrid &, const FLT *, UBIGINT, const FLT *, const FLT *, const FLT *,
    const BIGINT *, FLT *) const noexcept;

template int FINUFFT_PLAN_T<FLT>::spreadSorted(FLT *, const FLT *, int) const;
template int FINUFFT_PLAN_T<FLT>::interpSorted(FLT *, FLT *, int) const;
template void FINUFFT_PLAN_T<FLT>::indexSort();
template FLT FINUFFT_PLAN_T<FLT>::evaluate_kernel_runtime(FLT) const;
