#include <finufft/spread.hpp>

// Explicit instantiation of 1D spread subproblem, selected by FINUFFT_SINGLE define.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

template void FINUFFT_PLAN_T<FLT>::spread_subproblem_1d(BIGINT, UBIGINT, FLT *, UBIGINT,
                                                         FLT *, FLT *) const noexcept;
