#include <cuda.h>
#include <helper_cuda.h>
#include <iostream>
#include <iomanip>

#include <cuComplex.h>
#include "precision_independent.h"
#include "common.h"

using namespace std;

/* Kernel for computing approximations of exact Fourier series coeffs of
   cnufftspread's real symmetric kernel. */
// a , f are intermediate results from function onedim_fseries_kernel_precomp()
// (see cufinufft/contrib/common.cpp for description)
__global__
void FseriesKernelCompute(int nf1, int nf2, int nf3, FLT *f, cuDoubleComplex *a,
	FLT *fwkerhalf1, FLT *fwkerhalf2, FLT *fwkerhalf3, int ns)
{
	FLT J2 = ns/2.0;
	int q=(int)(2 + 3.0*J2);
	int nf;
	cuDoubleComplex *at = a + threadIdx.y*MAX_NQUAD;
	FLT *ft = f + threadIdx.y*MAX_NQUAD;
	FLT *oarr;
	if (threadIdx.y == 0){
		oarr = fwkerhalf1;
		nf = nf1;
	}else if (threadIdx.y == 1){
		oarr = fwkerhalf2;
		nf = nf2;
	}else{
		oarr = fwkerhalf3;
		nf = nf3;
	}

	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<nf/2+1; i+=blockDim.x*gridDim.x){
		int brk = 0.5 + i;
		FLT x = 0.0;
		for(int n=0; n<q; n++){
			x += ft[n] * 2*(pow(cabs(at[n]), brk)*cos(brk*carg(at[n])));
		}
		oarr[i] = x;
	}
}

int CUFSERIESKERNELCOMPUTE(int dim, int nf1, int nf2, int nf3, FLT *d_f,
	cuDoubleComplex *d_a, FLT *d_fwkerhalf1, FLT *d_fwkerhalf2,
	FLT *d_fwkerhalf3, int ns)
/*
	wrapper for approximation of Fourier series of real symmetric spreading 
	kernel.

	Melody Shih 2/20/22
*/
{
	int nout = max(max(nf1/2+1,nf2/2+1),nf3/2+1);

	dim3 threadsPerBlock(16, dim);
	dim3 numBlocks((nout+16-1)/16, 1);

	FseriesKernelCompute<<<numBlocks, threadsPerBlock>>>(nf1, nf2, nf3, d_f,
		d_a, d_fwkerhalf1, d_fwkerhalf2, d_fwkerhalf3, ns);
	return 0;
}
