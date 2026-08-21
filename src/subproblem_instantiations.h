// The per-dimension spread and interp subproblem symbols. One dimension is defined in
// each of spreadinterp_1d/2d/3d.cpp, and declared extern wherever else the driver is
// instantiated, so the signature lives in one place.
#pragma once

#define FINUFFT_INSTANTIATE_SUBPROBLEMS(PREFIX, NDIMS)                              \
  PREFIX template void FINUFFT_PLAN_T<FLT>::spread_subproblem<NDIMS>(               \
      const Subgrid &, FLT *, UBIGINT, const FLT *, const FLT *, const FLT *,       \
      const FLT *) const noexcept;                                                  \
  PREFIX template void FINUFFT_PLAN_T<FLT>::interp_subproblem<NDIMS>(               \
      const Subgrid &, const FLT *, UBIGINT, const FLT *, const FLT *, const FLT *, \
      const BIGINT *, FLT *) const noexcept;
