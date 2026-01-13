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

#ifndef SPREADINTERP_H
#define SPREADINTERP_H

#include <finufft/finufft_core.h>
#include <finufft_common/spread_opts.h>

namespace finufft {
namespace spreadinterp {

int spreadcheck(UBIGINT N1, UBIGINT N2, UBIGINT N3, const finufft_spread_opts &opts);
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

#endif // SPREADINTERP_H
