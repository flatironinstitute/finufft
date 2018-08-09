#include <iostream>
#include <math.h>
#include <helper_cuda.h>
#include <cuda.h>
#include "finufft/utils.h"
#include "spread.h"

using namespace std;

#define maxns 16

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static __inline__ __device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val +
					__longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif

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
#include "finufft/ker_horner_allw_loop.c"
	}
}

__global__
void Spread_2d_Idriven(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, int fw_width)
{
	int xstart,ystart,xend,yend;
	int xx, yy, ix, iy;
	int outidx;

	FLT x_rescaled, y_rescaled;
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		//x_rescaled = RESCALE(x[i],nf1,1);
		//y_rescaled = RESCALE(y[i],nf2,1);
		x_rescaled=x[i];
		y_rescaled=y[i];
		xstart = ceil(x_rescaled - ns/2.0);
		ystart = ceil(y_rescaled - ns/2.0);
		xend = floor(x_rescaled + ns/2.0);
		yend = floor(y_rescaled + ns/2.0);

		for(yy=ystart; yy<=yend; yy++){
			FLT disy=abs(y_rescaled-yy);
			FLT kervalue2 = evaluate_kernel(disy, es_c, es_beta);
			for(xx=xstart; xx<=xend; xx++){
				ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
				iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
				outidx = ix+iy*fw_width;
				FLT disx=abs(x_rescaled-xx);
				FLT kervalue1 = evaluate_kernel(disx, es_c, es_beta);
				atomicAdd(&fw[outidx].x, c[i].x*kervalue1*kervalue2);
				atomicAdd(&fw[outidx].y, c[i].y*kervalue1*kervalue2);
				//atomicAdd(&fw[outidx].x, kervalue1*kervalue2);
				//atomicAdd(&fw[outidx].y, kervalue1*kervalue2);
			}
		}

	}

}

__global__
void Spread_2d_Idriven_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, int fw_width)
{
	int xx, yy, ix, iy;
	int outidx;
	FLT ker1[7];
	//FLT ker2[7];
	FLT ker1val, ker2val;
	//double sigma=2.0;

	FLT x_rescaled, y_rescaled;
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		x_rescaled=x[i];
		y_rescaled=y[i];
		int xstart = ceil(x_rescaled - ns/2.0);
		int ystart = ceil(y_rescaled - ns/2.0);
		int xend = floor(x_rescaled + ns/2.0);
		int yend = floor(y_rescaled + ns/2.0);

		FLT x1=(FLT)xstart-x_rescaled;
		//FLT y1=(FLT)ystart-y_rescaled;
		//eval_kernel_vec_Horner(ker1,x1,ns,sigma);
		//eval_kernel_vec_Horner(ker2,y1,ns,sigma);
		evaluate_kernel_vector(ker1, x1, es_c, es_beta, ns);
		for(yy=ystart; yy<=yend; yy++){
			FLT disy=abs(y_rescaled-yy);
			FLT kervalue2 = evaluate_kernel(disy, es_c, es_beta);
			//ker2val=ker2[yy-ystart];
			for(xx=xstart; xx<=xend; xx++){
				ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
				iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
				outidx = ix+iy*fw_width;
				ker1val=ker1[xx-xstart];
				FLT kervalue=ker1val*ker2val;
				atomicAdd(&fw[outidx].x, c[i].x*kervalue);
				atomicAdd(&fw[outidx].y, c[i].y*kervalue);
			}
		}
	}
}

__global__
void CalcBinSize_noghost_2d(int M, int nf1, int nf2, int  bin_size_x, int bin_size_y, int nbinx,
		int nbiny, int* bin_size, FLT *x, FLT *y, int* sortidx)
{
	int binidx, binx, biny;
	int oldidx;
	FLT x_rescaled,y_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		//x_rescaled = RESCALE(x[i],nf1,1);
		//y_rescaled = RESCALE(y[i],nf2,1);
		x_rescaled=x[i];
		y_rescaled=y[i];
		binx = floor(x_rescaled/bin_size_x);
		biny = floor(y_rescaled/bin_size_y);
		binidx = binx+biny*nbinx;
		oldidx = atomicAdd(&bin_size[binidx], 1);
		sortidx[i] = oldidx;
	}
}

__global__
void PtsRearrage_noghost_2d(int M, int nf1, int nf2, int bin_size_x, int bin_size_y, int nbinx,
		int nbiny, int* bin_startpts, int* sortidx, FLT *x, FLT *x_sorted,
		FLT *y, FLT *y_sorted, CUCPX *c, CUCPX *c_sorted)
{
	//int i = blockDim.x*blockIdx.x + threadIdx.x;
	int binx, biny;
	int binidx;
	FLT x_rescaled, y_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		//x_rescaled = RESCALE(x[i],nf1,1);
		//y_rescaled = RESCALE(y[i],nf2,1);
		x_rescaled=x[i];
		y_rescaled=y[i];
		binx = floor(x_rescaled/bin_size_x);
		biny = floor(y_rescaled/bin_size_y);
		binidx = binx+biny*nbinx;
		
		x_sorted[bin_startpts[binidx]+sortidx[i]] = x_rescaled;
		y_sorted[bin_startpts[binidx]+sortidx[i]] = y_rescaled;
		c_sorted[bin_startpts[binidx]+sortidx[i]] = c[i];
	}
}

__global__
void CalcInvertofGlobalSortIdx_2d(int M, int bin_size_x, int bin_size_y, int nbinx,
			          int nbiny, int* bin_startpts, int* sortidx, 
                                  FLT *x, FLT *y, int* index)
{
	int binx, biny;
	int binidx;
	FLT x_rescaled, y_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=x[i];
		y_rescaled=y[i];
		binx = floor(x_rescaled/bin_size_x);
		biny = floor(y_rescaled/bin_size_y);
		binidx = binx+biny*nbinx;

		index[bin_startpts[binidx]+sortidx[i]] = i;
	}
}

__global__
void CalcSubProb_2d(int* bin_size, int* num_subprob, int maxsubprobsize, int numbins)
{
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<numbins; i+=gridDim.x*blockDim.x){
		num_subprob[i]=ceil(bin_size[i]/(float) maxsubprobsize);
	}
}

__global__
void MapBintoSubProb_2d(int* d_subprob_to_bin, int* d_subprobstartpts, int* d_numsubprob, 
                        int numbins)
{
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<numbins; i+=gridDim.x*blockDim.x){
		for(int j=0; j<d_numsubprob[i]; j++){
			d_subprob_to_bin[d_subprobstartpts[i]+j]=i;
		}
	}
}

__global__
void CreateSortIdx(int M, int nf1, int nf2, FLT *x, FLT *y, int* sortidx)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	FLT x_rescaled,y_rescaled;
	if (i < M){
		//x_rescaled = RESCALE(x[i],nf1,1);
		//y_rescaled = RESCALE(y[i],nf2,1);
		x_rescaled=x[i];
		y_rescaled=y[i];
		sortidx[i] = floor(x_rescaled) + floor(y_rescaled)*nf1;
	}
}

__global__
void Spread_2d_Simple(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		      int nf1, int nf2, FLT es_c, FLT es_beta, int fw_width, int bin_size, 
                      int bin_size_x, int bin_size_y, int binx, int biny)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,xend,yend;
	int xx, yy, ix, iy;
	int outidx;
	int ptstart=0;

	int xoffset=binx*bin_size_x;
	int yoffset=biny*bin_size_y;

	int N = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0));
	for(int i=threadIdx.x+threadIdx.y*blockDim.x; i<N; i+=blockDim.x*blockDim.y){
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();

	FLT x_rescaled, y_rescaled;
	for(int i=threadIdx.x+threadIdx.y*blockDim.x; i<bin_size; i+=blockDim.x*blockDim.y){
		int idx=ptstart+i;
		x_rescaled=x[idx];
		y_rescaled=y[idx];
		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		xend = floor(x_rescaled + ns/2.0)-xoffset;
		yend = floor(y_rescaled + ns/2.0)-yoffset;
		for(yy=ystart; yy<=yend; yy++){
			FLT disy=abs(y_rescaled-(yy+yoffset));
			FLT kervalue2 = evaluate_kernel(disy, es_c, es_beta);
			for(xx=xstart; xx<=xend; xx++){
				ix = xx+ceil(ns/2.0);
				iy = yy+ceil(ns/2.0);
				outidx = ix+iy*(bin_size_x+ceil(ns/2.0)*2);
				FLT disx=abs(x_rescaled-(xx+xoffset));
				FLT kervalue1 = evaluate_kernel(disx, es_c, es_beta);
				atomicAdd(&fwshared[outidx].x, c[idx].x*kervalue1*kervalue2);
				atomicAdd(&fwshared[outidx].y, c[idx].y*kervalue1*kervalue2);
			}
		}
	}
	__syncthreads();

	/* write to global memory */
	for(int k=threadIdx.x+threadIdx.y*blockDim.x; k<N; k+=blockDim.x*blockDim.y){
		int i = k % (int) (bin_size_x+2*ceil(ns/2.0) );
		int j = k /( bin_size_x+2*ceil(ns/2.0) );
		ix = xoffset+i-ceil(ns/2.0);
		iy = yoffset+j-ceil(ns/2.0);
		if(ix < (nf1+ceil(ns/2.0)) && iy < (nf2+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			iy = iy < 0 ? iy+nf2 : (iy>nf2-1 ? iy-nf2 : iy);
			outidx = ix+iy*fw_width;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2);
			atomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
			atomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
			//atomicAdd(&fw[outidx].x, y_rescaled);
			//atomicAdd(&fw[outidx].y, ystart+ceil(ns/2.0));
		}
	}
}

__global__
void Spread_2d_Hybrid(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, int fw_width, int* binstartpts,
		int* bin_size, int bin_size_x, int bin_size_y)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,xend,yend;
	int bidx=blockIdx.x+blockIdx.y*gridDim.x;
	int xx, yy, ix, iy;
	int outidx;
	int ptstart=binstartpts[bidx];

	int xoffset=blockIdx.x*bin_size_x;
	int yoffset=blockIdx.y*bin_size_y;

	int N = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0));
	for(int i=threadIdx.x+threadIdx.y*blockDim.x; i<N; i+=blockDim.x*blockDim.y){
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();

	FLT x_rescaled, y_rescaled;
	for(int i=threadIdx.x+threadIdx.y*blockDim.x; i<bin_size[bidx]; i+=blockDim.x*blockDim.y){
		int idx=ptstart+i;
		x_rescaled=x[idx];
		y_rescaled=y[idx];
		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		xend = floor(x_rescaled + ns/2.0)-xoffset;
		yend = floor(y_rescaled + ns/2.0)-yoffset;

		for(yy=ystart; yy<=yend; yy++){
			FLT disy=abs(y_rescaled-(yy+yoffset));
			FLT kervalue2 = evaluate_kernel(disy, es_c, es_beta);
			for(xx=xstart; xx<=xend; xx++){
				ix = xx+ceil(ns/2.0);
				iy = yy+ceil(ns/2.0);
				outidx = ix+iy*(bin_size_x+ceil(ns/2.0)*2);
				FLT disx=abs(x_rescaled-(xx+xoffset));
				FLT kervalue1 = evaluate_kernel(disx, es_c, es_beta);
				//fwshared[outidx].x += kervalue1*kervalue2;
				//fwshared[outidx].y += kervalue1*kervalue2;
				atomicAdd(&fwshared[outidx].x, c[idx].x*kervalue1*kervalue2);
				atomicAdd(&fwshared[outidx].y, c[idx].y*kervalue1*kervalue2);
				//atomicAdd(&fwshared[outidx].x, kervalue1*kervalue2);
				//atomicAdd(&fwshared[outidx].y, kervalue1*kervalue2);
			}
		}
	}
	__syncthreads();
	/* write to global memory */
	for(int k=threadIdx.x+threadIdx.y*blockDim.x; k<N; k+=blockDim.x*blockDim.y){
		int i = k % (int) (bin_size_x+2*ceil(ns/2.0) );
		int j = k /( bin_size_x+2*ceil(ns/2.0) );
		ix = xoffset+i-ceil(ns/2.0);
		iy = yoffset+j-ceil(ns/2.0);
		if(ix < (nf1+ceil(ns/2.0)) && iy < (nf2+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			iy = iy < 0 ? iy+nf2 : (iy>nf2-1 ? iy-nf2 : iy);
			outidx = ix+iy*fw_width;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2);
			atomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
			atomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
		}
	}
}

__global__
void Spread_2d_Subprob(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		          int nf1, int nf2, FLT es_c, FLT es_beta, FLT sigma, int fw_width, int* binstartpts,
		          int* bin_size, int bin_size_x, int bin_size_y, int* subprob_to_bin, 
		          int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx, int nbiny,
                          int* idxnupts)
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
	

	for(int i=threadIdx.x; i<N; i+=blockDim.x){
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();

	FLT x_rescaled, y_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		int idx = ptstart+i;
		x_rescaled = x[idxnupts[idx]];
		y_rescaled = y[idxnupts[idx]];
		cnow = c[idxnupts[idx]];

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;
		/*
		FLT ker1[maxns];
		FLT x1=(FLT) xstart+xoffset-x_rescaled;
        	for (int j = 0; j < ns; j++) { // Loop 1: Compute exponential arguments
                	ker1[j] = j;
        	}*/
		//evaluate_kernel_vector(ker1, x1, es_c, es_beta, ns);
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
				atomicAdd(&fwshared[outidx].x, cnow.x*kervalue1*kervalue2);
				atomicAdd(&fwshared[outidx].y, cnow.y*kervalue1*kervalue2);
			}
		}
	}
	__syncthreads();
	/* write to global memory */
	for(int k=threadIdx.x; k<N; k+=blockDim.x){
		int i = k % (int) (bin_size_x+2*ceil(ns/2.0) );
		int j = k /( bin_size_x+2*ceil(ns/2.0) );
		ix = xoffset-ceil(ns/2.0)+i;
		iy = yoffset-ceil(ns/2.0)+j;
		if(ix < (nf1+ceil(ns/2.0)) && iy < (nf2+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			iy = iy < 0 ? iy+nf2 : (iy>nf2-1 ? iy-nf2 : iy);
			outidx = ix+iy*fw_width;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2);
			atomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
			atomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
		}
	}
}
