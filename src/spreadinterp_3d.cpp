#include <finufft/spreadinterp.hpp>

#include "subproblem_instantiations.h"

// Per-dimension TU: explicit instantiation of 3D spread and interp for one precision.
// Compiled twice (with/without FINUFFT_SINGLE) to cover both float and double.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

FINUFFT_INSTANTIATE_SUBPROBLEMS(, 3)
