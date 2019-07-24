#include <iostream>
#include <math.h>
#include <helper_cuda.h>
#include <cuda.h>
#include "../../finufft/utils.h"
#include "../spreadinterp.h"

using namespace std;

static __forceinline__ __device__
FLT evaluate_kernel(FLT x, FLT es_c, FLT es_beta)
	/* ES ("exp sqrt") kernel evaluation at single real argument:
	   phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2
	   related to an asymptotic approximation to the Kaiser--Bessel, itself an
	   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
	   This is the "reference implementation", used by eg common/onedim_* 2/17/17 */
{
	return exp(es_beta * (sqrt(1.0 - es_c*x*x)));
	//return x;
	//return 1.0;
}
#if 0
static __forceinline__ __device__
void evaluate_kernel_vector(FLT *ker, FLT xstart, FLT es_c, FLT es_beta, const int N)
	/* Evaluate ES kernel for a vector of N arguments; by Ludvig af K.
	   If opts.kerpad true, args and ker must be allocated for Npad, and args is
	   written to (to pad to length Npad), only first N outputs are correct.
	   Barnett 4/24/18 option to pad to mult of 4 for better SIMD vectorization.
	   Obsolete (replaced by Horner), but keep around for experimentation since
	   works for arbitrary beta. Formula must match reference implementation. */
{
	// Note (by Ludvig af K): Splitting kernel evaluation into two loops
	// seems to benefit auto-vectorization.
	// gcc 5.4 vectorizes first loop; gcc 7.2 vectorizes both loops
	for (int i = 0; i < N; i++) { // Loop 1: Compute exponential arguments
		ker[i] = exp(es_beta * sqrt(1.0 - es_c*(xstart+i)*(xstart+i)));
	}
	//for (int i = 0; i < Npad; i++) // Loop 2: Compute exponentials
		//ker[i] = exp(ker[i]);
}
#endif

static __inline__ __device__
void eval_kernel_vec_Horner(FLT *ker, const FLT x, const int w, const double upsampfac)
	/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
	   x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
	   This is the current evaluation method, since it's faster (except i7 w=16).
	   Two upsampfacs implemented. Params must match ref formula. Barnett 4/24/18 */
{
	FLT z = 2*x + w - 1.0;         // scale so local grid offset z in [-1,1]
	// insert the auto-generated code which expects z, w args, writes to ker...
	if (upsampfac==2.0) {     // floating point equality is fine here
#include "../../finufft/ker_horner_allw_loop.c"
	}
}

__global__
void Interp_2d_Idriven(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		       int nf1, int nf2, FLT es_c, FLT es_beta)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){

		FLT x_rescaled=x[i];
		FLT y_rescaled=y[i];
		int xstart = ceil(x_rescaled - ns/2.0);
		int ystart = ceil(y_rescaled - ns/2.0);
		int xend = floor(x_rescaled + ns/2.0);
		int yend = floor(y_rescaled + ns/2.0);
		CUCPX cnow;
		cnow.x = 0.0;
		cnow.y = 0.0;
		for(int yy=ystart; yy<=yend; yy++){
			FLT disy=abs(y_rescaled-yy);
			FLT kervalue2 = evaluate_kernel(disy, es_c, es_beta);
			for(int xx=xstart; xx<=xend; xx++){
				int ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
				int iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
				int inidx = ix+iy*nf1;
				FLT disx=abs(x_rescaled-xx);
				FLT kervalue1 = evaluate_kernel(disx, es_c, es_beta);
				cnow.x += fw[inidx].x*kervalue1*kervalue2;
				cnow.y += fw[inidx].y*kervalue1*kervalue2;
			}
		}
		c[i].x = cnow.x;
		c[i].y = cnow.y;
	}

}

__global__
void Interp_2d_Idriven_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, FLT sigma)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		FLT x_rescaled=x[i];
		FLT y_rescaled=y[i];

		int xstart = ceil(x_rescaled - ns/2.0);
		int ystart = ceil(y_rescaled - ns/2.0);
		int xend = floor(x_rescaled + ns/2.0);
		int yend = floor(y_rescaled + ns/2.0);

		CUCPX cnow;
		cnow.x = 0.0;
		cnow.y = 0.0;
		FLT ker1[MAX_NSPREAD];
		FLT ker2[MAX_NSPREAD];

		eval_kernel_vec_Horner(ker1,xstart-x_rescaled,ns,sigma);
        eval_kernel_vec_Horner(ker2,ystart-y_rescaled,ns,sigma);

		for(int yy=ystart; yy<=yend; yy++){
			FLT disy=abs(y_rescaled-yy);
			FLT kervalue2 = ker2[yy-ystart];
			for(int xx=xstart; xx<=xend; xx++){
				int ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
				int iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
				int inidx = ix+iy*nf1;
				FLT disx=abs(x_rescaled-xx);
				FLT kervalue1 = ker1[xx-xstart];
				cnow.x += fw[inidx].x*kervalue1*kervalue2;
				cnow.y += fw[inidx].y*kervalue1*kervalue2;
			}
		}
		c[i].x = cnow.x;
		c[i].y = cnow.y;
	}

}

__global__
void Interp_2d_Subprob(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
	int nf1, int nf2, FLT es_c, FLT es_beta, FLT sigma, int* binstartpts,
	int* bin_size, int bin_size_x, int bin_size_y, int* subprob_to_bin,
	int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx, 
	int nbiny, int* idxnupts)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,xend,yend;
	int subpidx=blockIdx.x;
	int bidx=subprob_to_bin[subpidx];
	int binsubp_idx=subpidx-subprobstartpts[bidx];
	int ix, iy;
	int outidx;
	int ptstart=binstartpts[bidx]+binsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, bin_size[bidx]-binsubp_idx*maxsubprobsize);

	int xoffset=(bidx % nbinx)*bin_size_x;
	int yoffset=(bidx / nbinx)*bin_size_y;
	int N = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0));

	for(int k=threadIdx.x;k<N; k+=blockDim.x){
		int i = k % (int) (bin_size_x+2*ceil(ns/2.0) );
		int j = k /( bin_size_x+2*ceil(ns/2.0) );
		ix = xoffset-ceil(ns/2.0)+i;
		iy = yoffset-ceil(ns/2.0)+j;
		if(ix < (nf1+ceil(ns/2.0)) && iy < (nf2+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			iy = iy < 0 ? iy+nf2 : (iy>nf2-1 ? iy-nf2 : iy);
			outidx = ix+iy*nf1;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2);
			fwshared[sharedidx].x = fw[outidx].x;
			fwshared[sharedidx].y = fw[outidx].y;
		}
	}
	__syncthreads();

	FLT x_rescaled, y_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		int idx = ptstart+i;
		x_rescaled = x[idxnupts[idx]];
		y_rescaled = y[idxnupts[idx]];
		cnow.x = 0.0;
		cnow.y = 0.0;

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;

		for(int yy=ystart; yy<=yend; yy++){
			FLT disy=abs(y_rescaled-(yy+yoffset));
			FLT kervalue2 = evaluate_kernel(disy, es_c, es_beta);
			for(int xx=xstart; xx<=xend; xx++){
				ix = xx+ceil(ns/2.0);
				iy = yy+ceil(ns/2.0);
				outidx = ix+iy*(bin_size_x+ceil(ns/2.0)*2);
				FLT disx=abs(x_rescaled-(xx+xoffset));
				//FLT kervalue1 = ker1[xx-xstart];
				FLT kervalue1 = evaluate_kernel(disx, es_c, es_beta);
				cnow.x += fwshared[outidx].x*kervalue1*kervalue2;
				cnow.y += fwshared[outidx].y*kervalue1*kervalue2;
			}
		}
		c[idxnupts[idx]] = cnow;
	}
}

__global__
void Interp_2d_Subprob_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, FLT sigma, int* binstartpts, int* bin_size, 
	int bin_size_x, int bin_size_y, int* subprob_to_bin, int* subprobstartpts, 
	int* numsubprob, int maxsubprobsize, int nbinx, int nbiny, int* idxnupts)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,xend,yend;
	int subpidx=blockIdx.x;
	int bidx=subprob_to_bin[subpidx];
	int binsubp_idx=subpidx-subprobstartpts[bidx];
	int ix, iy;
	int outidx;
	int ptstart=binstartpts[bidx]+binsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, bin_size[bidx]-binsubp_idx*maxsubprobsize);

	int xoffset=(bidx % nbinx)*bin_size_x;
	int yoffset=(bidx / nbinx)*bin_size_y;

	int N = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0));


	for(int k=threadIdx.x;k<N; k+=blockDim.x){
		int i = k % (int) (bin_size_x+2*ceil(ns/2.0) );
		int j = k /( bin_size_x+2*ceil(ns/2.0) );
		ix = xoffset-ceil(ns/2.0)+i;
		iy = yoffset-ceil(ns/2.0)+j;
		if(ix < (nf1+ceil(ns/2.0)) && iy < (nf2+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			iy = iy < 0 ? iy+nf2 : (iy>nf2-1 ? iy-nf2 : iy);
			outidx = ix+iy*nf1;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2);
			fwshared[sharedidx].x = fw[outidx].x;
			fwshared[sharedidx].y = fw[outidx].y;
		}
	}
	__syncthreads();

	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];

	FLT x_rescaled, y_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		int idx = ptstart+i;
		x_rescaled = x[idxnupts[idx]];
		y_rescaled = y[idxnupts[idx]];
		cnow.x = 0.0;
		cnow.y = 0.0;

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;

		eval_kernel_vec_Horner(ker1,xstart+xoffset-x_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker2,ystart+yoffset-y_rescaled,ns,sigma);
		
		for(int yy=ystart; yy<=yend; yy++){
			FLT disy=abs(y_rescaled-(yy+yoffset));
			FLT kervalue2 = ker2[yy-ystart];
			for(int xx=xstart; xx<=xend; xx++){
				ix = xx+ceil(ns/2.0);
				iy = yy+ceil(ns/2.0);
				outidx = ix+iy*(bin_size_x+ceil(ns/2.0)*2);
		
				FLT kervalue1 = ker1[xx-xstart];
				cnow.x += fwshared[outidx].x*kervalue1*kervalue2;
				cnow.y += fwshared[outidx].y*kervalue1*kervalue2;
			}
		}
		c[idxnupts[idx]] = cnow;
	}
}
