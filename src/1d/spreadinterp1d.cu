#include <iostream>
#include <math.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <thrust/extrema.h>
#include "../../contrib/utils.h"
#include "../../contrib/utils_fp.h"
#include "../cuspreadinterp.h"
#include "../../include/utils.h"

using namespace std;

/* ------------------------ 1d Spreading Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */

__global__
void Spread_1d_NUptsdriven(FLT *x, CUCPX *c, CUCPX *fw, int M, const int ns, 
	int nf1, FLT es_c, FLT es_beta, int *idxnupts, int pirange)
{
	int xstart,xend;
	int xx, ix;
	FLT ker1[MAX_NSPREAD];

	FLT x_rescaled;
	FLT kervalue1;
	CUCPX cnow;
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
		cnow = c[idxnupts[i]];

		xstart = ceil(x_rescaled - ns/2.0);
		xend  = floor(x_rescaled + ns/2.0);

		FLT x1=(FLT)xstart-x_rescaled;
		eval_kernel_vec(ker1,x1,ns,es_c,es_beta);
		for(xx=xstart; xx<=xend; xx++){
			ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
			kervalue1=ker1[xx-xstart];
			atomicAdd(&fw[ix].x, cnow.x*kervalue1);
			atomicAdd(&fw[ix].y, cnow.y*kervalue1);
		}
	}

}

__global__
void Spread_1d_NUptsdriven_Horner(FLT *x, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, FLT sigma, int* idxnupts, int pirange)
{
	int xx, ix;
	FLT ker1[MAX_NSPREAD];

	FLT x_rescaled;
	CUCPX cnow;
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
		cnow = c[idxnupts[i]];
		int xstart = ceil(x_rescaled - ns/2.0);
		int xend  = floor(x_rescaled + ns/2.0);

		FLT x1=(FLT)xstart-x_rescaled;
		eval_kernel_vec_Horner(ker1,x1,ns,sigma);
		for(xx=xstart; xx<=xend; xx++){
			ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
			FLT kervalue=ker1[xx-xstart];
			atomicAdd(&fw[ix].x, cnow.x*kervalue);
			atomicAdd(&fw[ix].y, cnow.y*kervalue);
		}
	}
}

/* Kernels for SubProb Method */
// SubProb properties
__global__
void CalcBinSize_noghost_1d(int M, int nf1, int  bin_size_x, int nbinx, 
	int* bin_size, FLT *x, int* sortidx, int pirange)
{
	int binx;
	int oldidx;
	FLT x_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=RESCALE(x[i], nf1, pirange);
		binx = floor(x_rescaled/bin_size_x);
		binx = binx >= nbinx ? binx-1 : binx;
		binx = binx < 0 ? 0 : binx;
		oldidx = atomicAdd(&bin_size[binx], 1);
		sortidx[i] = oldidx;
		if(binx >= nbinx){
			sortidx[i] = -binx;
		}
	}
}

__global__
void CalcInvertofGlobalSortIdx_1d(int M, int bin_size_x, int nbinx, 
	int* bin_startpts, int* sortidx, FLT *x, int* index, int pirange, int nf1)
{
	int binx;
	FLT x_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=RESCALE(x[i], nf1, pirange);
		binx = floor(x_rescaled/bin_size_x);
		binx = binx >= nbinx ? binx-1 : binx;
		binx = binx < 0 ? 0 : binx;

		index[bin_startpts[binx]+sortidx[i]] = i;
	}
}


__global__
void Spread_1d_Subprob(FLT *x, CUCPX *c, CUCPX *fw, int M, const int ns,
	int nf1, FLT es_c, FLT es_beta, FLT sigma, int* binstartpts,
	int* bin_size, int bin_size_x, int* subprob_to_bin,
	int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx, 
	int* idxnupts, int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,xend;
	int subpidx=blockIdx.x;
	int bidx=subprob_to_bin[subpidx];
	int binsubp_idx=subpidx-subprobstartpts[bidx];
	int ix;
	int ptstart=binstartpts[bidx]+binsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, bin_size[bidx]-binsubp_idx*maxsubprobsize);

	int xoffset=(bidx % nbinx)*bin_size_x;

	int N = (bin_size_x+2*ceil(ns/2.0));
	FLT ker1[MAX_NSPREAD];
	
	for(int i=threadIdx.x; i<N; i+=blockDim.x){
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();

	FLT x_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		int idx = ptstart+i;
		x_rescaled=RESCALE(x[idxnupts[idx]], nf1, pirange);
		cnow = c[idxnupts[idx]];

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		xend   = floor(x_rescaled + ns/2.0)-xoffset;

		FLT x1=(FLT)xstart+xoffset - x_rescaled;
		eval_kernel_vec(ker1,x1,ns,es_c,es_beta);

		for(int xx=xstart; xx<=xend; xx++){
			ix = xx+ceil(ns/2.0);
			if(ix >= (bin_size_x + (int) ceil(ns/2.0)*2) || ix<0) break;
			atomicAdd(&fwshared[ix].x, cnow.x*ker1[xx-xstart]);
			atomicAdd(&fwshared[ix].y, cnow.y*ker1[xx-xstart]);
		}
	}
	__syncthreads();
	/* write to global memory */
	for(int k=threadIdx.x; k<N; k+=blockDim.x){
		ix = xoffset-ceil(ns/2.0)+k;
		if(ix < (nf1+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			atomicAdd(&fw[ix].x, fwshared[k].x);
			atomicAdd(&fw[ix].y, fwshared[k].y);
		}
	}
}

__global__
void Spread_1d_Subprob_Horner(FLT *x, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, FLT sigma, int* binstartpts, int* bin_size, 
	int bin_size_x, int* subprob_to_bin, int* subprobstartpts, 
	int* numsubprob, int maxsubprobsize, int nbinx, int* idxnupts, int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,xend;
	int subpidx=blockIdx.x;
	int bidx=subprob_to_bin[subpidx];
	int binsubp_idx=subpidx-subprobstartpts[bidx];
	int ix;
	int ptstart=binstartpts[bidx]+binsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, bin_size[bidx]-binsubp_idx*maxsubprobsize);

	int xoffset=(bidx % nbinx)*bin_size_x;

	int N = (bin_size_x+2*ceil(ns/2.0));
	
	FLT ker1[MAX_NSPREAD];

	for(int i=threadIdx.x; i<N; i+=blockDim.x){
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();

	FLT x_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		int idx = ptstart+i;
		x_rescaled=RESCALE(x[idxnupts[idx]], nf1, pirange);
		cnow = c[idxnupts[idx]];

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		xend  = floor(x_rescaled + ns/2.0)-xoffset;

		eval_kernel_vec_Horner(ker1,xstart+xoffset-x_rescaled,ns,sigma);

		for(int xx=xstart; xx<=xend; xx++){
			ix = xx+ceil(ns/2.0);
			if(ix >= (bin_size_x + (int) ceil(ns/2.0)*2) || ix<0) break;
			atomicAdd(&fwshared[ix].x, cnow.x*ker1[xx-xstart]);
			atomicAdd(&fwshared[ix].y, cnow.y*ker1[xx-xstart]);
		}
	}
	__syncthreads();

	/* write to global memory */
	for(int k=threadIdx.x; k<N; k+=blockDim.x){
		ix = xoffset-ceil(ns/2.0)+k;
		if(ix < (nf1+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			atomicAdd(&fw[ix].x, fwshared[k].x);
			atomicAdd(&fw[ix].y, fwshared[k].y);
		}
	}
}

/* --------------------- 1d Interpolation Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */
__global__
void Interp_1d_NUptsdriven(FLT *x, CUCPX *c, CUCPX *fw, int M, const int ns,
		       int nf1, FLT es_c, FLT es_beta, int* idxnupts, int pirange)
{
	FLT ker1[MAX_NSPREAD];
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		FLT x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
        
		int xstart = ceil(x_rescaled - ns/2.0);
		int xend  = floor(x_rescaled + ns/2.0);
		CUCPX cnow;
		cnow.x = 0.0;
		cnow.y = 0.0;

		FLT x1=(FLT)xstart-x_rescaled;
		eval_kernel_vec(ker1,x1,ns,es_c,es_beta);
		for(int xx=xstart; xx<=xend; xx++){
			int ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
			FLT kervalue1 = ker1[xx-xstart];
			cnow.x += fw[ix].x*kervalue1;
			cnow.y += fw[ix].y*kervalue1;
		}
		c[idxnupts[i]].x = cnow.x;
		c[idxnupts[i]].y = cnow.y;
	}
}

__global__
void Interp_1d_NUptsdriven_Horner(FLT *x, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, FLT sigma, int* idxnupts, int pirange)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		FLT x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);

		int xstart = ceil(x_rescaled - ns/2.0);
		int xend  = floor(x_rescaled + ns/2.0);

		CUCPX cnow;
		cnow.x = 0.0;
		cnow.y = 0.0;
		FLT ker1[MAX_NSPREAD];

		eval_kernel_vec_Horner(ker1,xstart-x_rescaled,ns,sigma);

		for(int xx=xstart; xx<=xend; xx++){
			int ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
			cnow.x += fw[ix].x*ker1[xx-xstart];
			cnow.y += fw[ix].y*ker1[xx-xstart];
		}
		c[idxnupts[i]].x = cnow.x;
		c[idxnupts[i]].y = cnow.y;
	}

}
