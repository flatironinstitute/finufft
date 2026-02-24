#include <finufft/detail/spread.hpp>

// Explicit instantiation, selected by FINUFFT_SINGLE define.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

namespace finufft::spreadinterp {
template int spreadSorted<FLT>(
    const std::vector<BIGINT> &sort_indices, UBIGINT N1, UBIGINT N2, UBIGINT N3,
    FLT *FINUFFT_RESTRICT data_uniform, UBIGINT M, const FLT *FINUFFT_RESTRICT kx,
    const FLT *FINUFFT_RESTRICT ky, const FLT *FINUFFT_RESTRICT kz,
    const FLT *data_nonuniform, const finufft_spread_opts &opts, int did_sort,
    const FLT *horner_coeffs_ptr, int nc);
} // namespace finufft::spreadinterp
