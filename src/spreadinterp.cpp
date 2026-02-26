#include <finufft/spreadinterp.hpp>

// Per-dimension functions are explicitly instantiated in spreadinterp_1d/2d/3d.cpp.
// This TU handles the remaining per-precision symbols.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

// Suppress re-instantiation of the per-dim symbols defined elsewhere:
extern template void FINUFFT_PLAN_T<FLT>::spread_subproblem_1d(BIGINT, UBIGINT, FLT *,
                                                                UBIGINT, FLT *,
                                                                FLT *) const noexcept;
extern template void FINUFFT_PLAN_T<FLT>::spread_subproblem_2d(
    BIGINT, BIGINT, UBIGINT, UBIGINT, FLT *FINUFFT_RESTRICT, UBIGINT, const FLT *,
    const FLT *, const FLT *) const noexcept;
extern template void FINUFFT_PLAN_T<FLT>::spread_subproblem_3d(BIGINT, BIGINT, BIGINT,
                                                                UBIGINT, UBIGINT, UBIGINT,
                                                                FLT *, UBIGINT, FLT *, FLT *,
                                                                FLT *, FLT *) const noexcept;
extern template int FINUFFT_PLAN_T<FLT>::interpSorted_1d(FLT *, FLT *) const;
extern template int FINUFFT_PLAN_T<FLT>::interpSorted_2d(FLT *, FLT *) const;
extern template int FINUFFT_PLAN_T<FLT>::interpSorted_3d(FLT *, FLT *) const;

template int FINUFFT_PLAN_T<FLT>::spreadSorted(FLT *FINUFFT_RESTRICT, const FLT *) const;
template int FINUFFT_PLAN_T<FLT>::interpSorted(FLT *FINUFFT_RESTRICT,
                                               FLT *FINUFFT_RESTRICT) const;
template int FINUFFT_PLAN_T<FLT>::spreadinterpSorted(FLT *, FLT *, bool) const;
template void FINUFFT_PLAN_T<FLT>::indexSort();
template FLT FINUFFT_PLAN_T<FLT>::evaluate_kernel_runtime(FLT) const;
