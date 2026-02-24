#include <finufft/spreadinterp.hpp>

// Explicit instantiation, selected by FINUFFT_SINGLE define.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

// Suppress per-dimension methods compiled in spread_1d/2d/3d.cpp.
extern template void FINUFFT_PLAN_T<FLT>::spread_subproblem_1d(BIGINT, UBIGINT, FLT *,
                                                                UBIGINT, FLT *,
                                                                FLT *) const noexcept;
extern template void FINUFFT_PLAN_T<FLT>::spread_subproblem_2d(
    BIGINT, BIGINT, UBIGINT, UBIGINT, FLT *FINUFFT_RESTRICT, UBIGINT, const FLT *,
    const FLT *, const FLT *) const noexcept;
extern template void FINUFFT_PLAN_T<FLT>::spread_subproblem_3d(
    BIGINT, BIGINT, BIGINT, UBIGINT, UBIGINT, UBIGINT, FLT *, UBIGINT, FLT *, FLT *,
    FLT *, FLT *) const noexcept;
template int FINUFFT_PLAN_T<FLT>::spreadSorted(FLT *FINUFFT_RESTRICT, const FLT *) const;
template int FINUFFT_PLAN_T<FLT>::interpSorted(FLT *FINUFFT_RESTRICT,
                                               FLT *FINUFFT_RESTRICT) const;
template int FINUFFT_PLAN_T<FLT>::spreadinterpSorted(FLT *, FLT *, bool) const;
template void FINUFFT_PLAN_T<FLT>::indexSort();
template FLT FINUFFT_PLAN_T<FLT>::evaluate_kernel_runtime(FLT) const;
