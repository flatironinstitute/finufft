#include <finufft/spreadinterp.hpp>

// FINUFFT_DIM=1/2/3 (combined with FINUFFT_SINGLE): instantiate that
// dimension's spread_subproblem_Nd for one precision, giving 6 parallel objects.
// No FINUFFT_DIM: instantiate the remaining symbols per-precision,
// with extern template suppressing the per-dim ones.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

#if FINUFFT_DIM == 1
template void FINUFFT_PLAN_T<FLT>::spread_subproblem_1d(BIGINT, UBIGINT, FLT *, UBIGINT,
                                                         FLT *, FLT *) const noexcept;

#elif FINUFFT_DIM == 2
template void FINUFFT_PLAN_T<FLT>::spread_subproblem_2d(
    BIGINT, BIGINT, UBIGINT, UBIGINT, FLT *FINUFFT_RESTRICT, UBIGINT, const FLT *,
    const FLT *, const FLT *) const noexcept;

#elif FINUFFT_DIM == 3
template void FINUFFT_PLAN_T<FLT>::spread_subproblem_3d(BIGINT, BIGINT, BIGINT, UBIGINT,
                                                         UBIGINT, UBIGINT, FLT *, UBIGINT,
                                                         FLT *, FLT *, FLT *,
                                                         FLT *) const noexcept;

#else

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
template int FINUFFT_PLAN_T<FLT>::spreadSorted(FLT *FINUFFT_RESTRICT, const FLT *) const;
template int FINUFFT_PLAN_T<FLT>::interpSorted(FLT *FINUFFT_RESTRICT,
                                               FLT *FINUFFT_RESTRICT) const;
template int FINUFFT_PLAN_T<FLT>::spreadinterpSorted(FLT *, FLT *, bool) const;
template void FINUFFT_PLAN_T<FLT>::indexSort();
template FLT FINUFFT_PLAN_T<FLT>::evaluate_kernel_runtime(FLT) const;
#endif
