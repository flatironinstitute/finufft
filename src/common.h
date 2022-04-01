#ifndef __COMMON_H__
#define __COMMON_H__ 
#include <cufinufft_eitherprec.h>

__global__
void FseriesKernelCompute(int nf1, int nf2, int nf3, FLT *f, cuDoubleComplex *a, FLT *fwkerhalf1, FLT *fwkerhalf2, FLT *fwkerhalf3, int ns);

int CUFSERIESKERNELCOMPUTE(int dim, int nf1, int nf2, int nf3, FLT *d_f, cuDoubleComplex *d_a, FLT *d_fwkerhalf1, FLT *d_fwkerhalf2, FLT *d_fwkerhalf3, int ns);
#endif
