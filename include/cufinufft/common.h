#ifndef __COMMON_H__
#define __COMMON_H__
#include <cufinufft_eitherprec.h>

namespace cufinufft {
namespace common {
__global__ void FseriesKernelCompute(int nf1, int nf2, int nf3, CUFINUFFT_FLT *f, cuDoubleComplex *a,
                                     CUFINUFFT_FLT *fwkerhalf1, CUFINUFFT_FLT *fwkerhalf2, CUFINUFFT_FLT *fwkerhalf3,
                                     int ns);

int CUFSERIESKERNELCOMPUTE(int dim, int nf1, int nf2, int nf3, CUFINUFFT_FLT *d_f, cuDoubleComplex *d_a,
                           CUFINUFFT_FLT *d_fwkerhalf1, CUFINUFFT_FLT *d_fwkerhalf2, CUFINUFFT_FLT *d_fwkerhalf3,
                           int ns);

int setup_spreader_for_nufft(finufft_spread_opts &spopts, CUFINUFFT_FLT eps, cufinufft_opts opts);
void SET_NF_TYPE12(CUFINUFFT_BIGINT ms, cufinufft_opts opts, finufft_spread_opts spopts, CUFINUFFT_BIGINT *nf,
                   CUFINUFFT_BIGINT b);
void onedim_fseries_kernel(CUFINUFFT_BIGINT nf, CUFINUFFT_FLT *fwkerhalf, finufft_spread_opts opts);
void onedim_fseries_kernel_precomp(CUFINUFFT_BIGINT nf, CUFINUFFT_FLT *f, dcomplex *a, finufft_spread_opts opts);
void onedim_fseries_kernel_compute(CUFINUFFT_BIGINT nf, CUFINUFFT_FLT *f, dcomplex *a, CUFINUFFT_FLT *fwkerhalf,
                                   finufft_spread_opts opts);

} // namespace common
} // namespace cufinufft
#endif
