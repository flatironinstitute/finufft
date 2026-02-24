// Defines interface to spreading/interpolation code.

/* Devnotes: see finufft_core.h for definition of MAX_NSPREAD
    RESCALE macro moved to spreadinterp.cpp, 7/15/20.
    finufft_spread_opts renamed 6/7/22.
    Note as of v2.5 (Dec 2025):
    legacy TF_OMIT_* timing flags were removed. Timing helpers
    Note as of v2.5: legacy TF_OMIT_* timing flags were removed. Timing helpers
    previously controlled by these flags have been purged from the codebase.
    The kerevalmeth/kerpad knobs remain in the public API structs solely for
    ABI compatibility and are ignored by the implementation (Horner is always
    used).
    1/9/26: setup_spreadinterp() moved to finufft_core/common.
*/

#pragma once

#include <finufft/finufft_core.hpp>
#include <finufft_common/spread_opts.h>

#include <cstdio>

namespace finufft {
namespace spreadinterp {

inline int spreadcheck(UBIGINT N1, UBIGINT N2, UBIGINT N3, const finufft_spread_opts &opts)
/* This does just the input checking and reporting for the spreader.
   See spreadinterp() for input arguments and meaning of returned value.
   Split out by Melody Shih, Jun 2018. Finiteness chk Barnett 7/30/18.
   Marco Barbone 5.8.24 removed bounds check as new foldrescale is not limited to
   [-3pi,3pi)
*/
{
  // INPUT CHECKING & REPORTING .... cuboid not too small for spreading?
  UBIGINT minN = UBIGINT(2 * opts.nspread);
  if (N1 < minN || (N2 > 1 && N2 < minN) || (N3 > 1 && N3 < minN)) {
    fprintf(stderr,
            "%s error: one or more non-trivial box dims is less than 2.nspread!\n",
            __func__);
    return FINUFFT_ERR_SPREAD_BOX_SMALL;
  }
  if (opts.spread_direction != 1 && opts.spread_direction != 2) {
    fprintf(stderr, "%s error: opts.spread_direction must be 1 or 2!\n", __func__);
    return FINUFFT_ERR_SPREAD_DIR;
  }
  return 0;
}
template<typename T>
int indexSort(std::vector<BIGINT> &sort_indices, UBIGINT N1, UBIGINT N2, UBIGINT N3,
              UBIGINT N, const T *kx, const T *ky, const T *kz,
              const finufft_spread_opts &opts);
template<typename T>
int spreadinterpSorted(
    const std::vector<BIGINT> &sort_indices, const UBIGINT N1, const UBIGINT N2,
    const UBIGINT N3, T *data_uniform, const UBIGINT M, const T *FINUFFT_RESTRICT kx,
    const T *FINUFFT_RESTRICT ky, const T *FINUFFT_RESTRICT kz,
    T *FINUFFT_RESTRICT data_nonuniform, const finufft_spread_opts &opts, int did_sort,
    bool adjoint, const T *horner_coeffs, int nc);

template<typename T>
T evaluate_kernel_runtime(T x, int ns, int nc, const T *horner_coeffs,
                          const finufft_spread_opts &opts);

} // namespace spreadinterp
} // namespace finufft
