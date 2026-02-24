#include <finufft/detail/execute.hpp>

// Explicit instantiation, selected by FINUFFT_SINGLE define.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

template int FINUFFT_PLAN_T<FLT>::execute_internal(
    std::complex<FLT> *cj, std::complex<FLT> *fk, bool adjoint, int ntrans_actual,
    std::complex<FLT> *aligned_scratch, size_t scratch_size) const;
