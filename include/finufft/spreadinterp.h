// Defines interface to spreading/interpolation code.

// Devnotes: see defs.h for definition of MAX_NSPREAD (as of 9/24/18).
// RESCALE macro moved to spreadinterp.cpp, 7/15/20.
// finufft_spread_opts renamed 6/7/22.

#ifndef SPREADINTERP_H
#define SPREADINTERP_H

#include <finufft/defs.h>
#include <finufft_spread_opts.h>

/* Bitwise debugging timing flag (TF) defs; see finufft_spread_opts.flags.
    This is an unobtrusive way to determine the time contributions of the
    different components of spreading/interp by selectively leaving them out.
    For example, running the following two tests shows the effect of the exp()
    in the kernel evaluation (the last argument is the flag):
    > perftest/spreadtestnd 3 8e6 8e6 1e-6 1 0 0 1 0
    > perftest/spreadtestnd 3 8e6 8e6 1e-6 1 4 0 1 0
    NOTE: non-zero values are for experts only, since
    NUMERICAL OUTPUT MAY BE INCORRECT UNLESS finufft_spread_opts.flags=0 !
*/
#define TF_OMIT_WRITE_TO_GRID        1 // don't add subgrids to out grid (dir=1)
#define TF_OMIT_EVALUATE_KERNEL      2 // don't evaluate the kernel at all
#define TF_OMIT_EVALUATE_EXPONENTIAL 4 // omit exp() in kernel (kereval=0 only)
#define TF_OMIT_SPREADING            8 // don't interp/spread (dir=1: to subgrids)

namespace finufft {
namespace spreadinterp {

// things external (spreadinterp) interface needs...
FINUFFT_EXPORT int FINUFFT_CDECL spreadinterp(
    UBIGINT N1, UBIGINT N2, UBIGINT N3, FLT *data_uniform, UBIGINT N, FLT *kx, FLT *ky,
    FLT *kz, FLT *data_nonuniform, const finufft_spread_opts &opts);
FINUFFT_EXPORT int FINUFFT_CDECL spreadcheck(UBIGINT N1, UBIGINT N2, UBIGINT N3,
                                             UBIGINT N, FLT *kx, FLT *ky, FLT *kz,
                                             const finufft_spread_opts &opts);
FINUFFT_EXPORT int FINUFFT_CDECL indexSort(BIGINT *sort_indices, UBIGINT N1, UBIGINT N2,
                                           UBIGINT N3, UBIGINT N, FLT *kx, FLT *ky,
                                           FLT *kz, const finufft_spread_opts &opts);
FINUFFT_EXPORT int FINUFFT_CDECL interpSorted(
    const BIGINT *sort_indices, UBIGINT N1, UBIGINT N2, UBIGINT N3,
    FLT *FINUFFT_RESTRICT data_uniform, UBIGINT N, FLT *FINUFFT_RESTRICT kx,
    FLT *FINUFFT_RESTRICT ky, FLT *FINUFFT_RESTRICT kz,
    FLT *FINUFFT_RESTRICT data_nonuniform, const finufft_spread_opts &opts);
FINUFFT_EXPORT int FINUFFT_CDECL spreadSorted(
    const BIGINT *sort_indices, UBIGINT N1, UBIGINT N2, UBIGINT N3, FLT *data_uniform,
    UBIGINT N, FLT *kx, FLT *ky, FLT *kz, const FLT *data_nonuniform,
    const finufft_spread_opts &opts, int did_sort);
FINUFFT_EXPORT int FINUFFT_CDECL spreadinterpSorted(
    const BIGINT *sort_indices, UBIGINT N1, UBIGINT N2, UBIGINT N3,
    FLT *FINUFFT_RESTRICT data_uniform, UBIGINT N, FLT *FINUFFT_RESTRICT kx,
    FLT *FINUFFT_RESTRICT ky, FLT *FINUFFT_RESTRICT kz,
    FLT *FINUFFT_RESTRICT data_nonuniform, const finufft_spread_opts &opts, int did_sort);
FINUFFT_EXPORT FLT FINUFFT_CDECL evaluate_kernel(FLT x, const finufft_spread_opts &opts);
FINUFFT_EXPORT int FINUFFT_CDECL setup_spreader(finufft_spread_opts &opts, FLT eps,
                                                double upsampfac, int kerevalmeth,
                                                int debug, int showwarn, int dim);

} // namespace spreadinterp
} // namespace finufft

#endif // SPREADINTERP_H
