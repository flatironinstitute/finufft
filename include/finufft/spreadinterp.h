// Defines interface to spreading/interpolation code.

// Devnotes: see finufft_core.h for definition of MAX_NSPREAD
// RESCALE macro moved to spreadinterp.cpp, 7/15/20.
// finufft_spread_opts renamed 6/7/22.

#ifndef SPREADINTERP_H
#define SPREADINTERP_H

#include <finufft/finufft_core.h>
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
enum {
  TF_OMIT_WRITE_TO_GRID        = 1, // don't add subgrids to out grid (dir=1)
  TF_OMIT_EVALUATE_KERNEL      = 2, // don't evaluate the kernel at all
  TF_OMIT_EVALUATE_EXPONENTIAL = 4, // omit exp() in kernel (kereval=0 only)
  TF_OMIT_SPREADING            = 8  // don't interp/spread (dir=1: to subgrids)
};

namespace finufft {
namespace spreadinterp {

// things external (spreadinterp) interface needs...
template<typename T>
FINUFFT_EXPORT_TEST int spreadinterp(UBIGINT N1, UBIGINT N2, UBIGINT N3, T *data_uniform,
                                     UBIGINT M, const T *kx, const T *ky, const T *kz,
                                     T *data_nonuniform, const finufft_spread_opts &opts);
template<typename T>
FINUFFT_EXPORT_TEST int setup_spreader(finufft_spread_opts &opts, T eps, double upsampfac,
                                       int kerevalmeth, int debug, int showwarn, int dim,
                                       int spreadinterponly);

int spreadcheck(UBIGINT N1, UBIGINT N2, UBIGINT N3, const finufft_spread_opts &opts);
template<typename T>
int indexSort(std::vector<BIGINT> &sort_indices, UBIGINT N1, UBIGINT N2, UBIGINT N3,
              UBIGINT N, const T *kx, const T *ky, const T *kz,
              const finufft_spread_opts &opts);
template<typename T>
int spreadinterpSorted(const std::vector<BIGINT> &sort_indices, const UBIGINT N1,
                       const UBIGINT N2, const UBIGINT N3, T *data_uniform,
                       const UBIGINT M, const T *FINUFFT_RESTRICT kx,
                       const T *FINUFFT_RESTRICT ky, const T *FINUFFT_RESTRICT kz,
                       T *FINUFFT_RESTRICT data_nonuniform,
                       const finufft_spread_opts &opts, int did_sort, bool adjoint);
template<typename T> T evaluate_kernel(T x, const finufft_spread_opts &opts);
template<typename T> T evaluate_kernel_horner(T x, const finufft_spread_opts &opts);

} // namespace spreadinterp
} // namespace finufft

#endif // SPREADINTERP_H
