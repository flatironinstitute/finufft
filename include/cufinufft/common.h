#ifndef __COMMON_H__
#define __COMMON_H__
#include <cufinufft_eitherprec.h>

__global__ void FseriesKernelCompute(int nf1, int nf2, int nf3, CUFINUFFT_FLT *f, cuDoubleComplex *a,
                                     CUFINUFFT_FLT *fwkerhalf1, CUFINUFFT_FLT *fwkerhalf2, CUFINUFFT_FLT *fwkerhalf3,
                                     int ns);

int CUFSERIESKERNELCOMPUTE(int dim, int nf1, int nf2, int nf3, CUFINUFFT_FLT *d_f, cuDoubleComplex *d_a,
                           CUFINUFFT_FLT *d_fwkerhalf1, CUFINUFFT_FLT *d_fwkerhalf2, CUFINUFFT_FLT *d_fwkerhalf3,
                           int ns);

int setup_spreader_for_nufft(SPREAD_OPTS &spopts, CUFINUFFT_FLT eps, cufinufft_opts opts);
void SET_NF_TYPE12(CUFINUFFT_BIGINT ms, cufinufft_opts opts, SPREAD_OPTS spopts, CUFINUFFT_BIGINT *nf,
                   CUFINUFFT_BIGINT b);
void onedim_fseries_kernel(CUFINUFFT_BIGINT nf, CUFINUFFT_FLT *fwkerhalf, SPREAD_OPTS opts);
void onedim_fseries_kernel_precomp(CUFINUFFT_BIGINT nf, CUFINUFFT_FLT *f, dcomplex *a, SPREAD_OPTS opts);
void onedim_fseries_kernel_compute(CUFINUFFT_BIGINT nf, CUFINUFFT_FLT *f, dcomplex *a, CUFINUFFT_FLT *fwkerhalf,
                                   SPREAD_OPTS opts);
#endif
