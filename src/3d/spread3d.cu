#include <iostream>
#include <math.h>
#include <helper_cuda.h>
#include <cuda.h>
#include "../../finufft/utils.h"
#include "../spreadinterp.h"

using namespace std;

#define RESCALE(x,N,p) (p ? \
                       ((x*M_1_2PI + (x<-PI ? 1.5 : (x>PI ? -0.5 : 0.5)))*N) : \
                       (x<0 ? x+N : (x>N ? x-N : x)))


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
#if 0 
static __forceinline__ __device__
void evaluate_kernel_vector(FLT *ker, FLT xstart, FLT es_c, FLT es_beta, 
	const int N)
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
void eval_kernel_vec_Horner(FLT *ker, const FLT x, const int w, 
	const double upsampfac)
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
#if 0
__device__
int CalcGlobalIdx(int xidx, int yidx, int zidx, int onx, int ony, int onz, 
	int bnx, int bny, int bnz){
	int oix,oiy,oiz,b;
	oix = xidx/bnx;
	oiy = yidx/bny;
	oiz = zidx/bnz;
	return = (oix + oiy*onx + oiz*ony*onz)*(bnx*bny*bnz) + 
			 (xidx%bnx+yidx%bny*bnx+zidx%bnz*bny*bnx);
}
#endif
__device__
int CalcGlobalIdx(int xidx, int yidx, int zidx, int nbinx, int nbiny, int nbinz){
	return xidx + yidx*nbinx + zidx*nbinx*nbiny;
}


__global__
void RescaleXY_3d(int M, int nf1, int nf2, int nf3, FLT* x, FLT* y, FLT* z)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		x[i] = RESCALE(x[i], nf1, 1);
		y[i] = RESCALE(y[i], nf2, 1);
		z[i] = RESCALE(z[i], nf3, 1);
	}
}

__global__
void Spread_3d_Idriven_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, FLT es_c, FLT es_beta)
{
	int xx, yy, ix, iy;
	int outidx;
	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];
	FLT ker1val, ker2val;
	double sigma=2.0;

	FLT x_rescaled, y_rescaled;
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		x_rescaled=x[i];
		y_rescaled=y[i];
		int xstart = ceil(x_rescaled - ns/2.0);
		int ystart = ceil(y_rescaled - ns/2.0);
		int xend = floor(x_rescaled + ns/2.0);
		int yend = floor(y_rescaled + ns/2.0);

		FLT x1=(FLT)xstart-x_rescaled;
		FLT y1=(FLT)ystart-y_rescaled;
		eval_kernel_vec_Horner(ker1,x1,ns,sigma);
		eval_kernel_vec_Horner(ker2,y1,ns,sigma);
		//evaluate_kernel_vector(ker1, x1, es_c, es_beta, ns);
		//evaluate_kernel_vector(ker2, y1, es_c, es_beta, ns);
		for(yy=ystart; yy<=yend; yy++){
			for(xx=xstart; xx<=xend; xx++){
				ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
				iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
				outidx = ix+iy*nf1;
				ker1val=ker1[xx-xstart];
				ker2val=ker2[yy-ystart];
				FLT kervalue=ker1val*ker2val;
				atomicAdd(&fw[outidx].x, c[i].x*kervalue);
				atomicAdd(&fw[outidx].y, c[i].y*kervalue);
			}
		}
	}
}

__global__
void LocateNUptstoBins(int M, int nf1, int nf2, int nf3, int  bin_size_x, 
	int bin_size_y, int bin_size_z, int nbinx, int nbiny, int nbinz, 
	int* bin_size, FLT *x, FLT *y, FLT *z, int* sortidx)
{
	int binidx,binx,biny,binz;
	int oldidx;
	FLT x_rescaled,y_rescaled,z_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=x[i];
		y_rescaled=y[i];
		z_rescaled=z[i];
		binx = floor(x_rescaled/bin_size_x);
		biny = floor(y_rescaled/bin_size_y);
		binz = floor(z_rescaled/bin_size_z);
		binidx = binx+biny*nbinx+binz*nbinx*nbiny;
		oldidx = atomicAdd(&bin_size[binidx], 1);
		sortidx[i] = oldidx;
	}
}

__global__
void LocateNUptstoBins_ghost(int M, int  bin_size_x, int bin_size_y, 
	int bin_size_z, int nbinx, int nbiny, int nbinz, int binsperobinx, 
	int binsperobiny, int binsperobinz, int* bin_size, FLT *x, FLT *y, FLT *z, 
	int* sortidx)
{
	int binidx,binx,biny,binz;
	int oldidx;
	FLT x_rescaled,y_rescaled,z_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=x[i];
		y_rescaled=y[i];
		z_rescaled=z[i];
		binx = floor(x_rescaled/bin_size_x);
		biny = floor(y_rescaled/bin_size_y);
		binz = floor(z_rescaled/bin_size_z);
		binx = binx/binsperobinx*(binsperobinx+2) + (binx%binsperobinx+1);
		biny = biny/binsperobiny*(binsperobiny+2) + (biny%binsperobiny+1);
		binz = binz/binsperobinz*(binsperobinz+2) + (binz%binsperobinz+1);

		binidx = CalcGlobalIdx(binx,biny,binz,nbinx,nbiny,nbinz);
		oldidx = atomicAdd(&bin_size[binidx], 1);
		sortidx[i] = oldidx;
	}
}

__global__
void Temp(int binsperobinx, int binsperobiny, int binsperobinz, 
	int nbinx, int nbiny, int nbinz, int* binsize)
{
	int binx =threadIdx.x+blockIdx.x*blockDim.x;
	int biny =threadIdx.y+blockIdx.y*blockDim.y;
	int binz =threadIdx.z+blockIdx.z*blockDim.z;
	int binidx = CalcGlobalIdx(binx, biny, binz, nbinx, nbiny, nbinz);
	
	if(binx < nbinx && biny < nbiny && binz < nbinz)
		if (binx%(binsperobinx+2) >=1 && binx%(binsperobinx+2)<= binsperobinx)
			if (biny%(binsperobiny+2) >=1 && biny%(binsperobiny+2) <= binsperobiny)
				if (binz%(binsperobinz+2) >=1 && binz%(binsperobinz+2)<= binsperobinz)
					binsize[binidx] = binidx;
}
__global__
void FillGhostBins(int binsperobinx, int binsperobiny, int binsperobinz, 
	int nbinx, int nbiny, int nbinz, int* binsize)
{
	int binx =threadIdx.x+blockIdx.x*blockDim.x;
	int biny =threadIdx.y+blockIdx.y*blockDim.y;
	int binz =threadIdx.z+blockIdx.z*blockDim.z;

	if(binx < nbinx && biny < nbiny && binz < nbinz){
		int binidx = CalcGlobalIdx(binx, biny, binz, nbinx, nbiny, nbinz);
		if(binx % (binsperobinx+2) == 1){
			int i = binx - 2;
			i = i<0 ? i+nbinx : i; 
			int idxtoupdate = CalcGlobalIdx(i, biny, binz, nbinx, nbiny, nbinz);
			binsize[idxtoupdate] = binsize[binidx]; 
		}
		if(binx % (binsperobinx+2) == binsperobinx){
			int i = binx + 2;
			i = (i==nbinx) ? i-nbinx : i; 
			int idxtoupdate = CalcGlobalIdx(i, biny, binz, nbinx, nbiny, nbinz);
			binsize[idxtoupdate] = binsize[binidx]; 
		}
		if(biny % (binsperobiny+2) == 1){
			int i = biny - 2;
			i = i<0 ? i+nbiny : i; 
			int idxtoupdate = CalcGlobalIdx(binx, i, binz, nbinx, nbiny, nbinz);
			binsize[idxtoupdate] =  binsize[binidx]; 
		}
		if(biny % (binsperobinx+2) == binsperobiny){
			int i = biny + 2;
			i = (i==nbiny) ? i-nbiny : i; 
			int idxtoupdate = CalcGlobalIdx(binx, i, binz, nbinx, nbiny, nbinz);
			binsize[idxtoupdate] = binsize[binidx];; 
		}
		if(binz % (binsperobinz+2) == 1){
			int i = binz - 2;
			i = i<0 ? i+nbinz : i; 
			int idxtoupdate = CalcGlobalIdx(binx, biny, i, nbinx, nbiny, nbinz);
			binsize[idxtoupdate] = binsize[binidx]; 
		}
		if(binz % (binsperobinz+2) == binsperobinz){
			int i = binz + 2;
			i = (i==nbinz) ? i-nbinz : i; 
			int idxtoupdate = CalcGlobalIdx(binx, biny, i, nbinx, nbiny, nbinz);
			binsize[idxtoupdate] = binsize[binidx]; 
		}
	}
}
__global__
void CalcInvertofGlobalSortIdx_ghost(int M, int  bin_size_x, 
	int bin_size_y, int bin_size_z, int nbinx, int nbiny, int nbinz, 
	int binsperobinx, int binsperobiny, int binsperobinz, int* bin_startpts, 
	int* sortidx, FLT *x, FLT *y, FLT *z, int* index)
{
	int binx,biny,binz;
	int binidx;
	FLT x_rescaled,y_rescaled,z_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=x[i];
		y_rescaled=y[i];
		z_rescaled=z[i];
		binx = floor(x_rescaled/bin_size_x);
		biny = floor(y_rescaled/bin_size_y);
		binz = floor(z_rescaled/bin_size_z);
		binx = binx/binsperobinx*(binsperobinx+2) + (binx%binsperobinx+1);
		biny = biny/binsperobiny*(binsperobiny+2) + (biny%binsperobiny+1);
		binz = binz/binsperobinz*(binsperobinz+2) + (binz%binsperobinz+1);

		binidx = CalcGlobalIdx(binx,biny,binz,nbinx,nbiny,nbinz);
		index[bin_startpts[binidx]+sortidx[i]] = i;
	}
}

__global__
void CalcInvertofGlobalSortIdx_3d(int M, int bin_size_x, int bin_size_y, 
	int bin_size_z, int nbinx, int nbiny, int nbinz, int* bin_startpts, 
	int* sortidx, FLT *x, FLT *y, FLT *z, int* index)
{
	int binx,biny,binz;
	int binidx;
	FLT x_rescaled,y_rescaled,z_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=x[i];
		y_rescaled=y[i];
		z_rescaled=z[i];
		binx = floor(x_rescaled/bin_size_x);
		biny = floor(y_rescaled/bin_size_y);
		binz = floor(z_rescaled/bin_size_z);
		binidx = CalcGlobalIdx(binx,biny,binz,nbinx,nbiny,nbinz);

		index[bin_startpts[binidx]+sortidx[i]] = i;
	}
}
__global__
void GhostBinPtsIdx(int binsperobinx, int binsperobiny, int binsperobinz, 
	int nbinx, int nbiny, int nbinz, int* binsize, int* index, 
	int* binstartpts)
{
	int binx =threadIdx.x+blockIdx.x*blockDim.x;
	int biny =threadIdx.y+blockIdx.y*blockDim.y;
	int binz =threadIdx.z+blockIdx.z*blockDim.z;
	int i,j,k;
	int w = 0;
	if(binx < nbinx && biny < nbiny && binz < nbinz){
		i = binx;
		j = biny;
		k = binz;
		int binidx = CalcGlobalIdx(binx, biny, binz, nbinx, nbiny, nbinz);
		if(binx % (binsperobinx+2) == 0){
			i = binx - 2;
			i = i<0 ? i+nbinx : i; 
			w=1;
		}
		if(binx % (binsperobinx+2) == binsperobinx+1){
			i = binx + 2;
			i = (i>nbinx) ? i-nbinx : i; 
			w=1;
		}
		if(biny % (binsperobiny+2) == 0){
			j = biny - 2;
			j = j<0 ? j+nbiny : j; 
			w=1;
		}
		if(biny % (binsperobiny+2) == binsperobiny+1){
			j = biny + 2;
			j = (j>nbiny) ? j-nbiny : j; 
			w=1;
		}
		if(binz % (binsperobinz+2) == 0){
			k = binz - 2;
			k = k<0 ? k+nbinz : k; 
			w=1;
		}
		if(binz % (binsperobinz+2) == binsperobinz+1){
			k = binz + 2;
			k = (k>nbinz) ? k-nbinz : k; 
			w=1;
		}
		int corbinidx = CalcGlobalIdx(i,j,k,nbinx,nbiny,nbinz);
		if(w==1){
			for(int n = 0; n<binsize[binidx];n++){
				index[binstartpts[binidx]+n] = index[binstartpts[corbinidx]+n];
			}
		}
	}
	
}

__global__
void CalcSubProb_3d_v1(int* bin_size, int* num_subprob, int maxsubprobsize, 
	int numbins)
{
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<numbins; i+=gridDim.x*
		blockDim.x){
		num_subprob[i]=ceil(bin_size[i]/(float) maxsubprobsize);
	}
}


// This kernels assumes that number of bins less than #maxnumberofthreads in 
// each dim

__global__
void CalcSubProb_3d(int bin_size_x, int bin_size_y, int bin_size_z, 
	int o_bin_size_x, int o_bin_size_y, int o_bin_size_z, int nbinx, int nbiny, 
	int nbinz, int nobinx, int nobiny, int nobinz, int* bin_size, 
	int* num_subprob, int* num_nupts, int maxsubprobsize)
{
	int numNUpts = 0;
	int numSubProbs = 0;
	int xbinstart, xbinend, ybinstart, ybinend, zbinstart, zbinend;

	int xobin, yobin, zobin;
	xobin = threadIdx.x+blockIdx.x*blockDim.x;
	yobin = threadIdx.y+blockIdx.y*blockDim.y;
	zobin = threadIdx.z+blockIdx.z*blockDim.z;
	
	int nbins_obin_x, nbins_obin_y, nbins_obin_z;
	nbins_obin_x = o_bin_size_x/bin_size_x;
	nbins_obin_y = o_bin_size_y/bin_size_y;
	nbins_obin_z = o_bin_size_z/bin_size_z;

	if(xobin < nobinx && yobin < nobiny && zobin < nobinz){
		xbinstart = xobin*nbins_obin_x-1;
		xbinend  = (xobin+1)*nbins_obin_x;
		ybinstart = yobin*nbins_obin_y-1;
		ybinend  = (yobin+1)*nbins_obin_y;
		zbinstart = zobin*nbins_obin_z-1;
		zbinend  = (zobin+1)*nbins_obin_z;

		int ix, iy, iz;
		for(int k = zbinstart; k<= zbinend; k++){
			iz = (k < 0)      ? k + nbinz : k; 
			iz = (k == nbinz) ? k - nbinz : iz; 
			for(int j=ybinstart; j<= ybinend; j++){
				iy = (j < 0)      ? j + nbiny : j; 
				iy = (j == nbiny) ? j - nbiny : iy; 
				for(int i=xbinstart; i<= xbinend; i++){
					ix = (i < 0)      ? i + nbinx : i; 
					ix = (i == nbinx) ? i - nbinx : ix; 
					int binidx = ix+iy*nbinx+iz*nbiny*nbinx;
					numNUpts += bin_size[binidx];
					//numSubProbs += ceil(bin_size[binidx]/ 
					//(float) maxsubprobsize);
				}
			}
		}
		int obinidx = xobin + yobin*nobinx + zobin*nobiny*nobinx;
		num_subprob[obinidx] = ceil(numNUpts/ (float) maxsubprobsize);
		//num_subprob[obinidx] = numSubProbs;
		num_nupts[obinidx]   = numNUpts;
	}
}

__global__
void MapBintoSubProb_3d(int* d_subprobstartpts, int* d_subprob_to_bin, 
	int* d_subprob_to_nupts, int bin_size_x, int bin_size_y, int bin_size_z, 
	int o_bin_size_x, int o_bin_size_y, int o_bin_size_z, int nbinx, 
	int nbiny, int nbinz, int nobinx, int nobiny, int nobinz, int* bin_size, 
	int* num_subprob, int* num_nupts, int maxsubprobsize)
{
	int numNUpts = 0;
	int s = 0;
	int xbinstart, xbinend, ybinstart, ybinend, zbinstart, zbinend;

	int xobin, yobin, zobin;
	xobin = threadIdx.x+blockIdx.x*blockDim.x;
	yobin = threadIdx.y+blockIdx.y*blockDim.y;
	zobin = threadIdx.z+blockIdx.z*blockDim.z;
	
	int nbins_obin_x, nbins_obin_y, nbins_obin_z;
	nbins_obin_x = o_bin_size_x/bin_size_x;
	nbins_obin_y = o_bin_size_y/bin_size_y;
	nbins_obin_z = o_bin_size_z/bin_size_z;

	if(xobin < nobinx && yobin < nobiny && zobin < nobinz){
		int obinidx = xobin + yobin*nobinx + zobin*nobiny*nobinx;
		int startsubprob = d_subprobstartpts[obinidx];
		int totalnupts = num_nupts[obinidx];

		xbinstart = xobin*nbins_obin_x-1;
		xbinend  = (xobin+1)*nbins_obin_x;
		ybinstart = yobin*nbins_obin_y-1;
		ybinend  = (yobin+1)*nbins_obin_y;
		zbinstart = zobin*nbins_obin_z-1;
		zbinend  = (zobin+1)*nbins_obin_z;
		
		int ix, iy, iz;
		for(int k = zbinstart; k<= zbinend; k++){
			iz = (k < 0)      ? k + nbinz : k; 
			iz = (iz == nbinz) ? iz - nbinz : iz; 
			for(int j=ybinstart; j<= ybinend; j++){
				iy = (j < 0)      ? j + nbiny : j; 
				iy = (iy == nbiny) ? iy - nbiny : iy; 
				for(int i=xbinstart; i<= xbinend; i++){
					ix = (i < 0)      ? i + nbinx : i; 
					ix = (ix == nbinx) ? ix - nbinx : ix;
					int binidx = ix+iy*nbinx+iz*nbiny*nbinx;
					int numNUptsold = numNUpts - maxsubprobsize;
					numNUpts += bin_size[binidx];
					if(s == 0 && numNUpts > 0){
						numNUptsold += maxsubprobsize;
						d_subprob_to_bin[startsubprob+s] = binidx;
						d_subprob_to_nupts[startsubprob+s] = 0;
						s++;
					}
					while( numNUpts >= maxsubprobsize ){
						numNUptsold += maxsubprobsize;
						d_subprob_to_bin  [startsubprob+s] = binidx;
						d_subprob_to_nupts[startsubprob+s] = numNUptsold;
						numNUpts -= maxsubprobsize;
						s++;
					}
				}
			}
		}
	}
}

__global__
void Spread_3d_Subprob(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
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
	
	//FLT ker1[MAX_NSPREAD];
	//FLT ker2[MAX_NSPREAD];


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

		for(int yy=ystart; yy<=yend; yy++){
			FLT disy=abs(y_rescaled-(yy+yoffset));
			FLT kervalue2 = evaluate_kernel(disy, es_c, es_beta);
			for(int xx=xstart; xx<=xend; xx++){
				ix = xx+ceil(ns/2.0);
				iy = yy+ceil(ns/2.0);
				outidx = ix+iy*(bin_size_x+ceil(ns/2.0)*2);
				FLT disx=abs(x_rescaled-(xx+xoffset));
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
			outidx = ix+iy*nf1;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2);
			atomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
			atomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
		}
	}
}

__global__
void Spread_3d_Subprob_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
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
	
	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];


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
			outidx = ix+iy*nf1;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2);
			atomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
			atomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
		}
	}
}
