#ifndef __COMMON_H__
#define __COMMON_H__

#include <finufft_spread_opts.h>
#include <cufinufft_types.h>
#include <cufinufft_opts.h>
#include <cufft.h>

namespace cufinufft {
namespace common {
template <typename T>
__global__ void FseriesKernelCompute(int nf1, int nf2, int nf3, T *f, cuDoubleComplex *a,
                                     T *fwkerhalf1, T *fwkerhalf2, T *fwkerhalf3,
                                     int ns);
template <typename T>
int cufserieskernelcompute(int dim, int nf1, int nf2, int nf3, T *d_f, cuDoubleComplex *d_a,
                           T *d_fwkerhalf1, T *d_fwkerhalf2, T *d_fwkerhalf3,
                           int ns);
template <typename T>
int setup_spreader_for_nufft(finufft_spread_opts &spopts, T eps, cufinufft_opts opts);

void set_nf_type12(CUFINUFFT_BIGINT ms, cufinufft_opts opts, finufft_spread_opts spopts, CUFINUFFT_BIGINT *nf,
                   CUFINUFFT_BIGINT b);
template <typename T>
void onedim_fseries_kernel(CUFINUFFT_BIGINT nf, T *fwkerhalf, finufft_spread_opts opts);
template <typename T>
void onedim_fseries_kernel_precomp(CUFINUFFT_BIGINT nf, T *f, dcomplex *a, finufft_spread_opts opts);
template <typename T>
void onedim_fseries_kernel_compute(CUFINUFFT_BIGINT nf, T *f, dcomplex *a, T *fwkerhalf,
                                   finufft_spread_opts opts);

} // namespace common
} // namespace cufinufft
#endif
