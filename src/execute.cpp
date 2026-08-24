#include <finufft/execute.hpp>

// Explicit instantiation, selected by FINUFFT_SINGLE define.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

// Suppress re-instantiation of all spread/interp method templates.
extern template int FINUFFT_PLAN_T<FLT>::spreadSorted(FLT *, const FLT *, int) const;
extern template int FINUFFT_PLAN_T<FLT>::interpSorted(FLT *, FLT *, int) const;
extern template void FINUFFT_PLAN_T<FLT>::spread_subproblem_1d(
    const Subgrid &, FLT *, UBIGINT, const FLT *, const FLT *, const FLT *,
    const FLT *) const noexcept;
extern template void FINUFFT_PLAN_T<FLT>::interp_subproblem_1d(
    const Subgrid &, const FLT *, UBIGINT, const FLT *, const FLT *, const FLT *,
    const BIGINT *, FLT *) const noexcept;
extern template void FINUFFT_PLAN_T<FLT>::spread_subproblem_2d(
    const Subgrid &, FLT *, UBIGINT, const FLT *, const FLT *, const FLT *,
    const FLT *) const noexcept;
extern template void FINUFFT_PLAN_T<FLT>::interp_subproblem_2d(
    const Subgrid &, const FLT *, UBIGINT, const FLT *, const FLT *, const FLT *,
    const BIGINT *, FLT *) const noexcept;
extern template void FINUFFT_PLAN_T<FLT>::spread_subproblem_3d(
    const Subgrid &, FLT *, UBIGINT, const FLT *, const FLT *, const FLT *,
    const FLT *) const noexcept;
extern template void FINUFFT_PLAN_T<FLT>::interp_subproblem_3d(
    const Subgrid &, const FLT *, UBIGINT, const FLT *, const FLT *, const FLT *,
    const BIGINT *, FLT *) const noexcept;

template int FINUFFT_PLAN_T<FLT>::execute_internal(
    std::complex<FLT> *cj, std::complex<FLT> *fk, bool adjoint, int ntrans_actual,
    std::complex<FLT> *aligned_scratch, size_t scratch_size) const;
