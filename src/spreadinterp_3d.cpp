#include <finufft/spreadinterp.hpp>

// Per-dimension TU: explicit instantiation of 3D spread and interp for one precision.
// Compiled twice (with/without FINUFFT_SINGLE) to cover both float and double.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

template void FINUFFT_PLAN_T<FLT>::spread_subproblem_3d(
    const Subgrid &, FLT *, UBIGINT, const FLT *, const FLT *, const FLT *,
    const FLT *) const noexcept;
template void FINUFFT_PLAN_T<FLT>::interp_subproblem_3d(
    const Subgrid &, const FLT *, UBIGINT, const FLT *, const FLT *, const FLT *,
    const BIGINT *, FLT *) const noexcept;
