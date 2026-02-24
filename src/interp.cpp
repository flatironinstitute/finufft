#include <finufft/detail/interp.hpp>

// Explicit instantiation, selected by FINUFFT_SINGLE define.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

namespace finufft::spreadinterp {
template int interpSorted<FLT>(
    const std::vector<BIGINT> &sort_indices, const UBIGINT N1, const UBIGINT N2,
    const UBIGINT N3, FLT *FINUFFT_RESTRICT data_uniform, const UBIGINT M,
    const FLT *FINUFFT_RESTRICT kx, const FLT *FINUFFT_RESTRICT ky,
    const FLT *FINUFFT_RESTRICT kz, FLT *FINUFFT_RESTRICT data_nonuniform,
    const finufft_spread_opts &opts, const FLT *horner_coeffs_ptr, int nc);
} // namespace finufft::spreadinterp
