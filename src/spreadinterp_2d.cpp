#include <finufft/spreadinterp.hpp>

// Per-dimension TU: explicit instantiation of 2D spread and interp for one precision.
// Compiled twice (with/without FINUFFT_SINGLE) to cover both float and double.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

template void FINUFFT_PLAN_T<FLT>::spread_subproblem_2d(
    BIGINT, BIGINT, UBIGINT, UBIGINT, FLT *FINUFFT_RESTRICT, UBIGINT, const FLT *,
    const FLT *, const FLT *) const noexcept;
template int FINUFFT_PLAN_T<FLT>::interpSorted_2d(FLT *, FLT *) const;
