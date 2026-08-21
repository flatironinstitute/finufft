#include <finufft/execute.hpp>

#include "subproblem_instantiations.h"

// Explicit instantiation, selected by FINUFFT_SINGLE define.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

// Suppress re-instantiation of all spread/interp method templates.
extern template int FINUFFT_PLAN_T<FLT>::spreadinterpTiled<1>(FLT *, FLT *, int) const;
extern template int FINUFFT_PLAN_T<FLT>::spreadinterpTiled<2>(FLT *, FLT *, int) const;
FINUFFT_INSTANTIATE_SUBPROBLEMS(extern, 1)
FINUFFT_INSTANTIATE_SUBPROBLEMS(extern, 2)
FINUFFT_INSTANTIATE_SUBPROBLEMS(extern, 3)

template int FINUFFT_PLAN_T<FLT>::execute_internal(
    std::complex<FLT> *cj, std::complex<FLT> *fk, bool adjoint, int ntrans_actual,
    std::complex<FLT> *aligned_scratch, size_t scratch_size) const;
