#include <finufft/spreadinterp.hpp>

// Per-dimension TU: explicit instantiation of 3D spread and interp for one precision.
// Compiled twice (with/without FINUFFT_SINGLE) to cover both float and double.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

template void FINUFFT_PLAN_T<FLT>::spread_subproblem_3d(
    BIGINT, BIGINT, BIGINT, UBIGINT, UBIGINT, UBIGINT, FLT *, UBIGINT, FLT *, FLT *,
    FLT *, FLT *) const noexcept;
template int FINUFFT_PLAN_T<FLT>::interpSorted_3d(FLT *, FLT *) const;
