#ifndef __COMMON_H__
#define __COMMON_H__ 
#include <cufinufft_eitherprec.h>

__global__
void FseriesKernelCompute(int nf1, int nf2, int nf3, CUFINUFFT_FLT *f, cuDoubleComplex *a, CUFINUFFT_FLT *fwkerhalf1, CUFINUFFT_FLT *fwkerhalf2, CUFINUFFT_FLT *fwkerhalf3, int ns);

int CUFSERIESKERNELCOMPUTE(int dim, int nf1, int nf2, int nf3, CUFINUFFT_FLT *d_f, cuDoubleComplex *d_a, CUFINUFFT_FLT *d_fwkerhalf1, CUFINUFFT_FLT *d_fwkerhalf2, CUFINUFFT_FLT *d_fwkerhalf3, int ns);
#endif
