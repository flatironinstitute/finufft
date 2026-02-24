#include <finufft/detail/spreadinterp.hpp>

// Explicit instantiation, selected by FINUFFT_SINGLE define.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

// Prevent re-instantiation of heavy spread/interp templates (compiled in
// spread.cpp and interp.cpp respectively).
namespace finufft::spreadinterp {
extern template int spreadSorted<FLT>(
    const std::vector<BIGINT> &, UBIGINT, UBIGINT, UBIGINT, FLT *FINUFFT_RESTRICT,
    UBIGINT, const FLT *FINUFFT_RESTRICT, const FLT *FINUFFT_RESTRICT,
    const FLT *FINUFFT_RESTRICT, const FLT *, const finufft_spread_opts &, int,
    const FLT *, int);
extern template int interpSorted<FLT>(
    const std::vector<BIGINT> &, const UBIGINT, const UBIGINT, const UBIGINT,
    FLT *FINUFFT_RESTRICT, const UBIGINT, const FLT *FINUFFT_RESTRICT,
    const FLT *FINUFFT_RESTRICT, const FLT *FINUFFT_RESTRICT, FLT *FINUFFT_RESTRICT,
    const finufft_spread_opts &, const FLT *, int);

template int spreadinterpSorted<FLT>(
    const std::vector<BIGINT> &sort_indices, const UBIGINT N1, const UBIGINT N2,
    const UBIGINT N3, FLT *data_uniform, const UBIGINT M,
    const FLT *FINUFFT_RESTRICT kx, const FLT *FINUFFT_RESTRICT ky,
    const FLT *FINUFFT_RESTRICT kz, FLT *FINUFFT_RESTRICT data_nonuniform,
    const finufft_spread_opts &opts, int did_sort, bool adjoint,
    const FLT *horner_coeffs, int nc);

template int indexSort<FLT>(std::vector<BIGINT> &sort_indices, UBIGINT N1, UBIGINT N2,
                            UBIGINT N3, UBIGINT M, const FLT *kx, const FLT *ky,
                            const FLT *kz, const finufft_spread_opts &opts);

template FLT evaluate_kernel_runtime<FLT>(FLT, int, int, const FLT *,
                                          const finufft_spread_opts &);
} // namespace finufft::spreadinterp
