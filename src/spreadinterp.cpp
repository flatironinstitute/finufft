#include <finufft/spreadinterp.hpp>

#include "subproblem_instantiations.h"

// Per-dimension functions are explicitly instantiated in spreadinterp_1d/2d/3d.cpp.
// This TU handles the remaining per-precision symbols.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

// Suppress re-instantiation of the per-dim symbols defined elsewhere:
FINUFFT_INSTANTIATE_SUBPROBLEMS(extern, 1)
FINUFFT_INSTANTIATE_SUBPROBLEMS(extern, 2)
FINUFFT_INSTANTIATE_SUBPROBLEMS(extern, 3)

template int FINUFFT_PLAN_T<FLT>::spreadinterpTiled<1>(FLT *, FLT *, int) const;
template int FINUFFT_PLAN_T<FLT>::spreadinterpTiled<2>(FLT *, FLT *, int) const;
template void FINUFFT_PLAN_T<FLT>::indexSort();
template FLT FINUFFT_PLAN_T<FLT>::evaluate_kernel_runtime(FLT) const;
