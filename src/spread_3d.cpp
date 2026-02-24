#include <finufft/spread.hpp>

// Explicit instantiation of 3D spread subproblem, selected by FINUFFT_SINGLE define.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

template void FINUFFT_PLAN_T<FLT>::spread_subproblem_3d(BIGINT, BIGINT, BIGINT, UBIGINT,
                                                         UBIGINT, UBIGINT, FLT *, UBIGINT,
                                                         FLT *, FLT *, FLT *,
                                                         FLT *) const noexcept;
