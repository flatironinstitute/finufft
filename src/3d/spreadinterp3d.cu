#include <iostream>
#include <math.h>
#include <helper_cuda.h>
#include <cuda.h>
#include "../../contrib/utils.h"
#include <cuspreadinterp.h>

using namespace std;

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
#include "../../contrib/ker_horner_allw_loop.c"
	}
}

static __inline__ __device__
void eval_kernel_vec(FLT *ker, const FLT x, const double w, const double es_c,
                     const double es_beta)
{
    for(int i=0; i<w; i++){
        ker[i] = evaluate_kernel(abs(x+i), es_c, es_beta);
    }
}

/* Common Kernels */
__device__
int CalcGlobalIdx(int xidx, int yidx, int zidx, int onx, int ony, int onz,
	int bnx, int bny, int bnz){
	int oix,oiy,oiz;
	oix = xidx/bnx;
	oiy = yidx/bny;
	oiz = zidx/bnz;
	return   (oix + oiy*onx + oiz*ony*onz)*(bnx*bny*bnz) +
			 (xidx%bnx+yidx%bny*bnx+zidx%bnz*bny*bnx);
}

__device__
int CalcGlobalIdx_V2(int xidx, int yidx, int zidx, int nbinx, int nbiny, int nbinz){
	return xidx + yidx*nbinx + zidx*nbinx*nbiny;
}

#if 0
__global__
void RescaleXY_3d(int M, int nf1, int nf2, int nf3, FLT* x, FLT* y, FLT* z)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		x[i] = RESCALE(x[i], nf1, 1);
		y[i] = RESCALE(y[i], nf2, 1);
		z[i] = RESCALE(z[i], nf3, 1);
	}
}
#endif
/* ---------------------- 3d Spreading Kernels -------------------------------*/
/* Kernels for bin sort NUpts */
__global__
void CalcBinSize_noghost_3d(int M, int nf1, int nf2, int nf3, int  bin_size_x,
	int bin_size_y, int bin_size_z, int nbinx, int nbiny, int nbinz,
    int* bin_size, FLT *x, FLT *y, FLT *z, int* sortidx, int pirange)
{
	int binidx, binx, biny, binz;
	int oldidx;
	FLT x_rescaled,y_rescaled,z_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=RESCALE(x[i], nf1, pirange);
		y_rescaled=RESCALE(y[i], nf2, pirange);
		z_rescaled=RESCALE(z[i], nf3, pirange);
		binx = floor(x_rescaled/bin_size_x);
		binx = binx >= nbinx ? binx-1 : binx;

		biny = floor(y_rescaled/bin_size_y);
		biny = biny >= nbiny ? biny-1 : biny;

		binz = floor(z_rescaled/bin_size_z);
		binz = binz >= nbinz ? binz-1 : binz;
		binidx = binx+biny*nbinx+binz*nbinx*nbiny;
		oldidx = atomicAdd(&bin_size[binidx], 1);
		sortidx[i] = oldidx;
	}
}

__global__
void CalcInvertofGlobalSortIdx_3d(int M, int bin_size_x, int bin_size_y,
	int bin_size_z, int nbinx, int nbiny, int nbinz, int* bin_startpts,
	int* sortidx, FLT *x, FLT *y, FLT *z, int* index, int pirange, int nf1, 
	int nf2, int nf3)
{
	int binx,biny,binz;
	int binidx;
	FLT x_rescaled,y_rescaled,z_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=RESCALE(x[i], nf1, pirange);
		y_rescaled=RESCALE(y[i], nf2, pirange);
		z_rescaled=RESCALE(z[i], nf3, pirange);
		binx = floor(x_rescaled/bin_size_x);
		binx = binx >= nbinx ? binx-1 : binx;
		biny = floor(y_rescaled/bin_size_y);
		biny = biny >= nbiny ? biny-1 : biny;
		binz = floor(z_rescaled/bin_size_z);
		binz = binz >= nbinz ? binz-1 : binz;
		binidx = CalcGlobalIdx_V2(binx,biny,binz,nbinx,nbiny,nbinz);

		index[bin_startpts[binidx]+sortidx[i]] = i;
	}
}

__global__ 
void TrivialGlobalSortIdx_3d(int M, int* index)
{
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		index[i] = i;
	}
}

/* Kernels for NUptsdriven method */
__global__
void Spread_3d_NUptsdriven_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, 
	int M, const int ns, int nf1, int nf2, int nf3, FLT sigma, int* idxnupts, 
	int pirange)
{
	int xx, yy, zz, ix, iy, iz;
	int outidx;
	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];
	FLT ker3[MAX_NSPREAD];

	FLT ker1val, ker2val, ker3val;

	FLT x_rescaled, y_rescaled, z_rescaled;
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
		y_rescaled=RESCALE(y[idxnupts[i]], nf2, pirange);
		z_rescaled=RESCALE(z[idxnupts[i]], nf3, pirange);

		int xstart = ceil(x_rescaled - ns/2.0);
		int ystart = ceil(y_rescaled - ns/2.0);
		int zstart = ceil(z_rescaled - ns/2.0);
		int xend = floor(x_rescaled + ns/2.0);
		int yend = floor(y_rescaled + ns/2.0);
		int zend = floor(z_rescaled + ns/2.0);

		FLT x1=(FLT)xstart-x_rescaled;
		FLT y1=(FLT)ystart-y_rescaled;
		FLT z1=(FLT)zstart-z_rescaled;

		eval_kernel_vec_Horner(ker1,x1,ns,sigma);
		eval_kernel_vec_Horner(ker2,y1,ns,sigma);
		eval_kernel_vec_Horner(ker3,z1,ns,sigma);
		for(zz=zstart; zz<=zend; zz++){
			ker3val=ker3[zz-zstart];
			for(yy=ystart; yy<=yend; yy++){
				ker2val=ker2[yy-ystart];
				for(xx=xstart; xx<=xend; xx++){
					ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
					iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
					iz = zz < 0 ? zz+nf3 : (zz>nf3-1 ? zz-nf3 : zz);
					outidx = ix+iy*nf1+iz*nf1*nf2;
					ker1val=ker1[xx-xstart];
					FLT kervalue=ker1val*ker2val*ker3val;
					atomicAdd(&fw[outidx].x, c[idxnupts[i]].x*kervalue);
					atomicAdd(&fw[outidx].y, c[idxnupts[i]].y*kervalue);
				}
			}
		}
	}
}
__global__
void Spread_3d_NUptsdriven(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta, 
	int* idxnupts, int pirange)
{
	int xx, yy, zz, ix, iy, iz;
	int outidx;
	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];
	FLT ker3[MAX_NSPREAD];

	FLT x_rescaled, y_rescaled, z_rescaled;
	FLT ker1val, ker2val, ker3val;
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
		y_rescaled=RESCALE(y[idxnupts[i]], nf2, pirange);
		z_rescaled=RESCALE(z[idxnupts[i]], nf3, pirange);

		int xstart = ceil(x_rescaled - ns/2.0);
		int ystart = ceil(y_rescaled - ns/2.0);
		int zstart = ceil(z_rescaled - ns/2.0);
		int xend = floor(x_rescaled + ns/2.0);
		int yend = floor(y_rescaled + ns/2.0);
		int zend = floor(z_rescaled + ns/2.0);

		FLT x1=(FLT)xstart-x_rescaled;
		FLT y1=(FLT)ystart-y_rescaled;
		FLT z1=(FLT)zstart-z_rescaled;

		eval_kernel_vec(ker1,x1,ns,es_c,es_beta);
		eval_kernel_vec(ker2,y1,ns,es_c,es_beta);
		eval_kernel_vec(ker3,z1,ns,es_c,es_beta);
		for(zz=zstart; zz<=zend; zz++){
			ker3val=ker3[zz-zstart];
			for(yy=ystart; yy<=yend; yy++){
				ker2val=ker2[yy-ystart];
				for(xx=xstart; xx<=xend; xx++){
					ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
					iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
					iz = zz < 0 ? zz+nf3 : (zz>nf3-1 ? zz-nf3 : zz);
					outidx = ix+iy*nf1+iz*nf1*nf2;

					ker1val=ker1[xx-xstart];
					FLT kervalue=ker1val*ker2val*ker3val;

					atomicAdd(&fw[outidx].x, c[idxnupts[i]].x*kervalue);
					atomicAdd(&fw[outidx].y, c[idxnupts[i]].y*kervalue);
				}
			}
		}
	}
}

/* Kernels for Subprob method */
__global__
void CalcSubProb_3d_v2(int* bin_size, int* num_subprob, int maxsubprobsize,
	int numbins)
{
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<numbins;
		i+=gridDim.x*blockDim.x){
		num_subprob[i]=ceil(bin_size[i]/(float) maxsubprobsize);
	}
}

__global__
void MapBintoSubProb_3d_v2(int* d_subprob_to_bin,int* d_subprobstartpts,
	int* d_numsubprob,int numbins)
{
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<numbins;
		i+=gridDim.x*blockDim.x){
		for(int j=0; j<d_numsubprob[i]; j++){
			d_subprob_to_bin[d_subprobstartpts[i]+j]=i;
		}
	}
}

__global__
void Spread_3d_Subprob_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT sigma, int* binstartpts,
	int* bin_size, int bin_size_x, int bin_size_y, int bin_size_z,
	int* subprob_to_bin, int* subprobstartpts, int* numsubprob, 
	int maxsubprobsize, int nbinx, int nbiny, int nbinz, int* idxnupts, 
	int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,xend,yend,zstart,zend;
	int bidx=subprob_to_bin[blockIdx.x];
	int binsubp_idx=blockIdx.x-subprobstartpts[bidx];
	int ix,iy,iz,outidx;
	int ptstart=binstartpts[bidx]+binsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, bin_size[bidx]-binsubp_idx*maxsubprobsize);

	int xoffset=(bidx % nbinx)*bin_size_x;
	int yoffset=((bidx / nbinx)%nbiny)*bin_size_y;
	int zoffset=(bidx/ (nbinx*nbiny))*bin_size_z;

	int N = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0))*
		(bin_size_z+2*ceil(ns/2.0));


	for(int i=threadIdx.x; i<N; i+=blockDim.x){
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();
	FLT x_rescaled, y_rescaled, z_rescaled;
	CUCPX cnow;

	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		FLT ker1[MAX_NSPREAD];
		FLT ker2[MAX_NSPREAD];
		FLT ker3[MAX_NSPREAD];
		
		int nuptsidx = idxnupts[ptstart+i];
		x_rescaled = RESCALE(x[nuptsidx],nf1,pirange);
		y_rescaled = RESCALE(y[nuptsidx],nf2,pirange);
		z_rescaled = RESCALE(z[nuptsidx],nf3,pirange);
		cnow = c[nuptsidx];

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		zstart = ceil(z_rescaled - ns/2.0)-zoffset;

		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;
		zend   = floor(z_rescaled + ns/2.0)-zoffset;

		eval_kernel_vec_Horner(ker1,xstart+xoffset-x_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker2,ystart+yoffset-y_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker3,zstart+zoffset-z_rescaled,ns,sigma);

    	for (int zz=zstart; zz<=zend; zz++){
			FLT kervalue3 = ker3[zz-zstart];
			iz = zz+ceil(ns/2.0);
			if(iz >= (bin_size_z + (int) ceil(ns/2.0)*2)) break;
			for(int yy=ystart; yy<=yend; yy++){
				FLT kervalue2 = ker2[yy-ystart];
				iy = yy+ceil(ns/2.0);
				if(iy >= (bin_size_y + (int) ceil(ns/2.0)*2)) break;
				for(int xx=xstart; xx<=xend; xx++){
					ix = xx+ceil(ns/2.0);
					if(ix >= (bin_size_x + (int) ceil(ns/2.0)*2)) break;
					outidx = ix+iy*(bin_size_x+ceil(ns/2.0)*2)+
						iz*(bin_size_x+ceil(ns/2.0)*2)*
						   (bin_size_y+ceil(ns/2.0)*2);
					FLT kervalue1 = ker1[xx-xstart];
					atomicAdd(&fwshared[outidx].x, cnow.x*kervalue1*kervalue2*
						kervalue3);
					atomicAdd(&fwshared[outidx].y, cnow.y*kervalue1*kervalue2*
						kervalue3);
        		}
      		}
		}
	}
	__syncthreads();
	/* write to global memory */
	for(int n=threadIdx.x; n<N; n+=blockDim.x){
		int i = n % (int) (bin_size_x+2*ceil(ns/2.0) );
		int j = (int) (n /(bin_size_x+2*ceil(ns/2.0))) % 
				(int) (bin_size_y+2*ceil(ns/2.0));
		int k = n / ((bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0)));

		ix = xoffset-ceil(ns/2.0)+i;
		iy = yoffset-ceil(ns/2.0)+j;
		iz = zoffset-ceil(ns/2.0)+k;

		if(ix<(nf1+ceil(ns/2.0)) && 
		   iy<(nf2+ceil(ns/2.0)) && 
		   iz<(nf3+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			iy = iy < 0 ? iy+nf2 : (iy>nf2-1 ? iy-nf2 : iy);
			iz = iz < 0 ? iz+nf3 : (iz>nf3-1 ? iz-nf3 : iz);
			outidx = ix+iy*nf1+iz*nf1*nf2;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2)+
				k*(bin_size_x+ceil(ns/2.0)*2)*(bin_size_y+ceil(ns/2.0)*2);
			atomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
			atomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
		}
	}
}

__global__
void Spread_3d_Subprob(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta, int* binstartpts,
	int* bin_size, int bin_size_x, int bin_size_y, int bin_size_z,
	int* subprob_to_bin, int* subprobstartpts, int* numsubprob, int maxsubprobsize,
	int nbinx, int nbiny, int nbinz, int* idxnupts, int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,xend,yend,zstart,zend;
	int subpidx=blockIdx.x;
	int bidx=subprob_to_bin[subpidx];
	int binsubp_idx=subpidx-subprobstartpts[bidx];
	int ix, iy, iz, outidx;
	int ptstart=binstartpts[bidx]+binsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, bin_size[bidx]-binsubp_idx*maxsubprobsize);

	int xoffset=(bidx % nbinx)*bin_size_x;
	int yoffset=((bidx / nbinx)%nbiny)*bin_size_y;
	int zoffset=(bidx/ (nbinx*nbiny))*bin_size_z;

	int N = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0))*
		(bin_size_z+2*ceil(ns/2.0));

	for(int i=threadIdx.x; i<N; i+=blockDim.x){
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();
	FLT x_rescaled, y_rescaled, z_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		FLT ker1[MAX_NSPREAD];
		FLT ker2[MAX_NSPREAD];
		FLT ker3[MAX_NSPREAD];
		int idx = ptstart+i;
		x_rescaled=RESCALE(x[idxnupts[idx]], nf1, pirange);
		y_rescaled=RESCALE(y[idxnupts[idx]], nf2, pirange);
		z_rescaled=RESCALE(z[idxnupts[idx]], nf3, pirange);
		cnow = c[idxnupts[idx]];

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		zstart = ceil(z_rescaled - ns/2.0)-zoffset;

		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;
		zend   = floor(z_rescaled + ns/2.0)-zoffset;

		FLT x1=(FLT)xstart+xoffset-x_rescaled;
		FLT y1=(FLT)ystart+yoffset-y_rescaled;
		FLT z1=(FLT)zstart+zoffset-z_rescaled;

		eval_kernel_vec(ker1,x1,ns,es_c,es_beta);
		eval_kernel_vec(ker2,y1,ns,es_c,es_beta);
		eval_kernel_vec(ker3,z1,ns,es_c,es_beta);
#if 1
		for(int zz=zstart; zz<=zend; zz++){
			FLT kervalue3 = ker3[zz-zstart];
			iz = zz+ceil(ns/2.0);
			for(int yy=ystart; yy<=yend; yy++){
				FLT kervalue2 = ker2[yy-ystart];
				iy = yy+ceil(ns/2.0);
				for(int xx=xstart; xx<=xend; xx++){
					FLT kervalue1 = ker1[xx-xstart];
					ix = xx+ceil(ns/2.0);
					if(ix >= (bin_size_x + (int) ceil(ns/2.0)*2)) break;
					if(iy >= (bin_size_y + (int) ceil(ns/2.0)*2)) break;
					if(iz >= (bin_size_z + (int) ceil(ns/2.0)*2)) break;
					outidx = ix+iy*(bin_size_x+ceil(ns/2.0)*2)+
							 iz*(bin_size_x+ceil(ns/2.0)*2)*
						        (bin_size_y+ceil(ns/2.0)*2);
#if 1
					atomicAdd(&fwshared[outidx].x, cnow.x*kervalue1*kervalue2*
						kervalue3);
					atomicAdd(&fwshared[outidx].y, cnow.y*kervalue1*kervalue2*
						kervalue3);
#endif
				}
			}
		}
#endif
	}
	__syncthreads();
	/* write to global memory */
	for(int n=threadIdx.x; n<N; n+=blockDim.x){
		int i = n % (int) (bin_size_x+2*ceil(ns/2.0) );
		int j = (int) (n /(bin_size_x+2*ceil(ns/2.0))) % (int) (bin_size_y+2*ceil(ns/2.0));
		int k = n / ((bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0)));

		ix = xoffset-ceil(ns/2.0)+i;
		iy = yoffset-ceil(ns/2.0)+j;
		iz = zoffset-ceil(ns/2.0)+k;
		if(ix<(nf1+ceil(ns/2.0)) && iy<(nf2+ceil(ns/2.0)) && iz<(nf3+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			iy = iy < 0 ? iy+nf2 : (iy>nf2-1 ? iy-nf2 : iy);
			iz = iz < 0 ? iz+nf3 : (iz>nf3-1 ? iz-nf3 : iz);
			outidx = ix+iy*nf1+iz*nf1*nf2;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2)+
				k*(bin_size_x+ceil(ns/2.0)*2)*(bin_size_y+ceil(ns/2.0)*2);
			atomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
			atomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
		}
	}
}
/* Kernels for Block BlockGather Method */
__global__
void Temp(int binsperobinx, int binsperobiny, int binsperobinz,
	int nobinx, int nobiny, int nobinz, int* binsize)
{
	int binx =threadIdx.x+blockIdx.x*blockDim.x;
	int biny =threadIdx.y+blockIdx.y*blockDim.y;
	int binz =threadIdx.z+blockIdx.z*blockDim.z;
	int binidx = CalcGlobalIdx(binx,biny,binz,nobinx,nobiny,nobinz,binsperobinx,
			binsperobiny, binsperobinz);

	if(binx < nobinx*binsperobinx && biny < nobiny*binsperobiny &&
		binz < nobinz*binsperobinz)
		if (binx%binsperobinx >0 && binx%binsperobinx< binsperobinx-1)
			if (biny%binsperobiny >0 && biny%binsperobiny< binsperobiny-1)
				if (binz%binsperobinz >0 && binz%binsperobinz< binsperobinz-1)
					binsize[binidx] = binidx;
}

__global__
void LocateNUptstoBins_ghost(int M, int  bin_size_x, int bin_size_y,
	int bin_size_z, int nobinx, int nobiny, int nobinz, int binsperobinx,
	int binsperobiny, int binsperobinz, int* bin_size, FLT *x, FLT *y, FLT *z,
	int* sortidx, int pirange, int nf1, int nf2, int nf3)
{
	int binidx,binx,biny,binz;
	int oldidx;
	FLT x_rescaled,y_rescaled,z_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=RESCALE(x[i], nf1, pirange);
		y_rescaled=RESCALE(y[i], nf2, pirange);
		z_rescaled=RESCALE(z[i], nf3, pirange);
		binx = floor(x_rescaled/bin_size_x);
		biny = floor(y_rescaled/bin_size_y);
		binz = floor(z_rescaled/bin_size_z);
		binx = binx/(binsperobinx-2)*binsperobinx + (binx%(binsperobinx-2)+1);
		biny = biny/(binsperobiny-2)*binsperobiny + (biny%(binsperobiny-2)+1);
		binz = binz/(binsperobinz-2)*binsperobinz + (binz%(binsperobinz-2)+1);

		binidx = CalcGlobalIdx(binx,biny,binz,nobinx,nobiny,nobinz,binsperobinx,
			binsperobiny, binsperobinz);
		oldidx = atomicAdd(&bin_size[binidx], 1);
		sortidx[i] = oldidx;
	}
}

__global__
void FillGhostBins(int binsperobinx, int binsperobiny, int binsperobinz,
	int nobinx, int nobiny, int nobinz, int* binsize)
{
	int binx =threadIdx.x+blockIdx.x*blockDim.x;
	int biny =threadIdx.y+blockIdx.y*blockDim.y;
	int binz =threadIdx.z+blockIdx.z*blockDim.z;

	int nbinx = nobinx*binsperobinx;
	int nbiny = nobiny*binsperobiny;
	int nbinz = nobinz*binsperobinz;

	if(binx < nbinx && biny < nbiny && binz < nbinz){
		int binidx = CalcGlobalIdx(binx,biny,binz,nobinx,nobiny,nobinz,
			binsperobinx,binsperobiny, binsperobinz);
		if(binx % binsperobinx == 1){
			int i = binx - 2;
			i = i<0 ? i+nbinx : i;
			int idxtoupdate = CalcGlobalIdx(i,biny,binz,nobinx,nobiny,nobinz,
				binsperobinx,binsperobiny, binsperobinz);
			binsize[idxtoupdate] = binsize[binidx];
		}
		if(binx % binsperobinx == binsperobinx-2){
			int i = binx + 2;
			i = (i==nbinx) ? i-nbinx : i;
			int idxtoupdate = CalcGlobalIdx(i,biny,binz,nobinx,nobiny,nobinz,
				binsperobinx,binsperobiny, binsperobinz);
			binsize[idxtoupdate] = binsize[binidx];
		}
		if(biny % binsperobiny == 1){
			int i = biny - 2;
			i = i<0 ? i+nbiny : i;
			int idxtoupdate = CalcGlobalIdx(binx,i,binz,nobinx,nobiny,nobinz,
				binsperobinx,binsperobiny, binsperobinz);
			binsize[idxtoupdate] =  binsize[binidx];
		}
		if(biny % binsperobinx == binsperobiny-2){
			int i = biny + 2;
			i = (i==nbiny) ? i-nbiny : i;
			int idxtoupdate = CalcGlobalIdx(binx,i,binz,nobinx,nobiny,nobinz,
				binsperobinx,binsperobiny, binsperobinz);
			binsize[idxtoupdate] = binsize[binidx];
		}
		if(binz % binsperobinz == 1){
			int i = binz - 2;
			i = i<0 ? i+nbinz : i;
			int idxtoupdate = CalcGlobalIdx(binx,biny,i,nobinx,nobiny,nobinz,
				binsperobinx,binsperobiny, binsperobinz);
			binsize[idxtoupdate] = binsize[binidx];
		}
		if(binz % binsperobinz == binsperobinz-2){
			int i = binz + 2;
			i = (i==nbinz) ? i-nbinz : i;
			int idxtoupdate = CalcGlobalIdx(binx,biny,i,nobinx,nobiny,nobinz,
				binsperobinx,binsperobiny, binsperobinz);
			binsize[idxtoupdate] = binsize[binidx];
		}
	}
}

__global__
void CalcInvertofGlobalSortIdx_ghost(int M, int  bin_size_x,
	int bin_size_y, int bin_size_z, int nobinx, int nobiny, int nobinz,
	int binsperobinx, int binsperobiny, int binsperobinz, int* bin_startpts,
	int* sortidx, FLT *x, FLT *y, FLT *z, int* index, int pirange, int nf1,
	int nf2, int nf3)
{
	int binx,biny,binz;
	int binidx;
	FLT x_rescaled,y_rescaled,z_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=RESCALE(x[i], nf1, pirange);
		y_rescaled=RESCALE(y[i], nf2, pirange);
		z_rescaled=RESCALE(z[i], nf3, pirange);
		binx = floor(x_rescaled/bin_size_x);
		biny = floor(y_rescaled/bin_size_y);
		binz = floor(z_rescaled/bin_size_z);
		binx = binx/(binsperobinx-2)*binsperobinx + (binx%(binsperobinx-2)+1);
		biny = biny/(binsperobiny-2)*binsperobiny + (biny%(binsperobiny-2)+1);
		binz = binz/(binsperobinz-2)*binsperobinz + (binz%(binsperobinz-2)+1);

		binidx = CalcGlobalIdx(binx,biny,binz,nobinx,nobiny,nobinz,binsperobinx,
			binsperobiny, binsperobinz);
		index[bin_startpts[binidx]+sortidx[i]] = i;
	}
}

__global__
void GhostBinPtsIdx(int binsperobinx, int binsperobiny, int binsperobinz,
	int nobinx, int nobiny, int nobinz, int* binsize, int* index,
	int* binstartpts, int M)
{
	int binx =threadIdx.x+blockIdx.x*blockDim.x;
	int biny =threadIdx.y+blockIdx.y*blockDim.y;
	int binz =threadIdx.z+blockIdx.z*blockDim.z;
	int nbinx = nobinx*binsperobinx;
	int nbiny = nobiny*binsperobiny;
	int nbinz = nobinz*binsperobinz;

	int i,j,k;
	int w = 0;
	int box[3];
	if(binx < nbinx && biny < nbiny && binz < nbinz){
		box[0] = box[1] = box[2] = 0;
		i = binx;
		j = biny;
		k = binz;
		int binidx = CalcGlobalIdx(binx,biny,binz,nobinx,nobiny,nobinz,
			binsperobinx,binsperobiny,binsperobinz);
		if(binx % binsperobinx == 0){
			i = binx - 2;
			box[0] = (i<0);
			i = i<0 ? i+nbinx : i;
			w=1;
		}
		if(binx % binsperobinx == binsperobinx-1){
			i = binx + 2;
			box[0] = (i>nbinx)*2;
			i = (i>nbinx) ? i-nbinx : i;
			w=1;
		}
		if(biny % binsperobiny == 0){
			j = biny - 2;
			box[1] = (j<0);
			j = j<0 ? j+nbiny : j;
			w=1;
		}
		if(biny % binsperobiny == binsperobiny-1){
			j = biny + 2;
			box[1] = (j>nbiny)*2;
			j = (j>nbiny) ? j-nbiny : j;
			w=1;
		}
		if(binz % binsperobinz == 0){
			k = binz - 2;
			box[2] = (k<0);
			k = k<0 ? k+nbinz : k;
			w=1;
		}
		if(binz % binsperobinz == binsperobinz-1){
			k = binz + 2;
			box[2] = (k>nbinz)*2;
			k = (k>nbinz) ? k-nbinz : k;
			w=1;
		}
		int corbinidx = CalcGlobalIdx(i,j,k,nobinx,nobiny,nobinz,
			binsperobinx,binsperobiny, binsperobinz);
		if(w==1){
			for(int n = 0; n<binsize[binidx];n++){
				index[binstartpts[binidx]+n] = M*(box[0]+box[1]*3+box[2]*9) +
					index[binstartpts[corbinidx]+n];
			}
		}
	}

}

__global__
void CalcSubProb_3d_v1(int binsperobinx, int binsperobiny, int binsperobinz,
	int* bin_size, int* num_subprob, int maxsubprobsize, int numbins)
{
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<numbins; i+=gridDim.x*
		blockDim.x){
		int numnupts = 0;
		int binsperobin = binsperobinx*binsperobiny*binsperobinz;
		for(int b = 0; b<binsperobin; b++){
			numnupts += bin_size[binsperobin*i+b];
		}
		num_subprob[i]=ceil(numnupts/(float) maxsubprobsize);
	}
}

__global__
void MapBintoSubProb_3d_v1(int* d_subprob_to_obin, int* d_subprobstartpts,
	int* d_numsubprob,int numbins)
{
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<numbins;
		i+=gridDim.x*blockDim.x){
		for(int j=0; j<d_numsubprob[i]; j++){
			d_subprob_to_obin[d_subprobstartpts[i]+j]=i;
		}
	}
}

__global__
void Spread_3d_BlockGather(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta, FLT sigma,
	int* binstartpts, int obin_size_x, int obin_size_y, int obin_size_z,
	int binsperobin, int* subprob_to_bin, int* subprobstartpts,
	int maxsubprobsize, int nobinx, int nobiny, int nobinz, int* idxnupts,
	int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,zstart,xend,yend,zend;
	int subpidx=blockIdx.x;
	int obidx=subprob_to_bin[subpidx];
	int bidx = obidx*binsperobin;

	int obinsubp_idx=subpidx-subprobstartpts[obidx];
	int ix, iy, iz;
	int outidx;
	int ptstart=binstartpts[bidx]+obinsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, binstartpts[bidx+binsperobin]-binstartpts[bidx]
			-obinsubp_idx*maxsubprobsize);

	int xoffset=(obidx % nobinx)*obin_size_x;
	int yoffset=(obidx / nobinx)%nobiny*obin_size_y;
	int zoffset=(obidx / (nobinx*nobiny))*obin_size_z;

	int N = obin_size_x*obin_size_y*obin_size_z;

	for(int i=threadIdx.x; i<N; i+=blockDim.x){
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();
	FLT x_rescaled, y_rescaled, z_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		int idx = ptstart+i;
		int b = idxnupts[idx]/M;
		int box[3];
		for(int d=0;d<3;d++){
			box[d] = b%3;
			if(box[d] == 1)
				box[d] = -1;
			b=b/3;
		}
		int ii = idxnupts[idx]%M;
		x_rescaled = RESCALE(x[ii],nf1,pirange) + box[0]*nf1;
		y_rescaled = RESCALE(y[ii],nf2,pirange) + box[1]*nf2;
		z_rescaled = RESCALE(z[ii],nf3,pirange) + box[2]*nf3;
		cnow = c[ii];

#if 1
		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		xstart = xstart < 0 ? 0 : xstart;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		ystart = ystart < 0 ? 0 : ystart;
		zstart = ceil(z_rescaled - ns/2.0)-zoffset;
		zstart = zstart < 0 ? 0 : zstart;
		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		xend   = xend >= obin_size_x ? obin_size_x-1 : xend;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;
		yend   = yend >= obin_size_y ? obin_size_y-1 : yend;
		zend   = floor(z_rescaled + ns/2.0)-zoffset;
		zend   = zend >= obin_size_z ? obin_size_z-1 : zend;
#else
		xstart = 0;
		ystart = 0;
		zstart = 0;
		xend = ns;
		yend = ns;
		zend = ns;
#endif
		for(int zz=zstart; zz<=zend; zz++){
			FLT disz=abs(z_rescaled-(zz+zoffset));
			FLT kervalue3 = evaluate_kernel(disz, es_c, es_beta);
			//FLT kervalue3 = disz;
			for(int yy=ystart; yy<=yend; yy++){
				FLT disy=abs(y_rescaled-(yy+yoffset));
				FLT kervalue2 = evaluate_kernel(disy, es_c, es_beta);
				//FLT kervalue2 = disy;
				for(int xx=xstart; xx<=xend; xx++){
					outidx = xx+yy*obin_size_x+zz*obin_size_y*obin_size_x;
					FLT disx=abs(x_rescaled-(xx+xoffset));
					FLT kervalue1 = evaluate_kernel(disx, es_c, es_beta);
					//FLT kervalue1 = disx;
	//				fwshared[outidx].x += cnow.x*kervalue1*kervalue2*kervalue3;
	//				fwshared[outidx].y += cnow.y*kervalue1*kervalue2*kervalue3;
#if 1
					atomicAdd(&fwshared[outidx].x, cnow.x*kervalue1*kervalue2*
						kervalue3);
					atomicAdd(&fwshared[outidx].y, cnow.y*kervalue1*kervalue2*
						kervalue3);
#endif
				}
			}
		}
	}
	__syncthreads();
	/* write to global memory */
	for(int n=threadIdx.x; n<N; n+=blockDim.x){
		int i = n%obin_size_x;
		int j = (n/obin_size_x)%obin_size_y;
		int k = n/(obin_size_x*obin_size_y);

		ix = xoffset+i;
		iy = yoffset+j;
		iz = zoffset+k;
		outidx = ix+iy*nf1+iz*nf1*nf2;
		atomicAdd(&fw[outidx].x, fwshared[n].x);
		atomicAdd(&fw[outidx].y, fwshared[n].y);
	}
}

__global__
void Spread_3d_BlockGather_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta, FLT sigma,
	int* binstartpts, int obin_size_x, int obin_size_y, int obin_size_z,
	int binsperobin, int* subprob_to_bin, int* subprobstartpts,
	int maxsubprobsize, int nobinx, int nobiny, int nobinz, int* idxnupts,
	int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,zstart,xend,yend,zend;
	int subpidx=blockIdx.x;
	int obidx=subprob_to_bin[subpidx];
	int bidx = obidx*binsperobin;

	int obinsubp_idx=subpidx-subprobstartpts[obidx];
	int ix, iy, iz;
	int outidx;
	int ptstart=binstartpts[bidx]+obinsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, binstartpts[bidx+binsperobin]-binstartpts[bidx]
			-obinsubp_idx*maxsubprobsize);

	int xoffset=(obidx%nobinx)*obin_size_x;
	int yoffset=(obidx/nobinx)%nobiny*obin_size_y;
	int zoffset=(obidx/(nobinx*nobiny))*obin_size_z;

	int N = obin_size_x*obin_size_y*obin_size_z;

	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];
	FLT ker3[MAX_NSPREAD];

	for(int i=threadIdx.x; i<N; i+=blockDim.x){
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();

	FLT x_rescaled, y_rescaled, z_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		int nidx = idxnupts[ptstart+i];
		int b = nidx/M;
		int box[3];
		for(int d=0;d<3;d++){
			box[d] = b%3;
			if(box[d] == 1)
				box[d] = -1;
			b=b/3;
		}
		int ii = nidx%M;
		x_rescaled = RESCALE(x[ii],nf1,pirange) + box[0]*nf1;
		y_rescaled = RESCALE(y[ii],nf2,pirange) + box[1]*nf2;
		z_rescaled = RESCALE(z[ii],nf3,pirange) + box[2]*nf3;
		cnow = c[ii];

#if 0
		xstart = max((int)ceil(x_rescaled - ns/2.0)-xoffset, 0);
		//xstart = xstart < 0 ? 0 : xstart;
		ystart = max((int)ceil(y_rescaled - ns/2.0)-yoffset, 0);
		//ystart = ystart < 0 ? 0 : ystart;
		zstart = max((int)ceil(z_rescaled - ns/2.0)-zoffset, 0);
		//zstart = zstart < 0 ? 0 : zstart;
		xend   = min((int)floor(x_rescaled + ns/2.0)-xoffset, obin_size_x-1);
		//xend   = xend >= obin_size_x ? obin_size_x-1 : xend;
		yend   = min((int)floor(y_rescaled + ns/2.0)-yoffset, obin_size_y-1);
		//yend   = yend >= obin_size_y ? obin_size_y-1 : yend;
		zend   = min((int)floor(z_rescaled + ns/2.0)-zoffset, obin_size_z-1);
		//zend   = zend >= obin_size_z ? obin_size_z-1 : zend;
#else
		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		xstart = xstart < 0 ? 0 : xstart;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		ystart = ystart < 0 ? 0 : ystart;
		zstart = ceil(z_rescaled - ns/2.0)-zoffset;
		zstart = zstart < 0 ? 0 : zstart;
		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		xend   = xend >= obin_size_x ? obin_size_x-1 : xend;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;
		yend   = yend >= obin_size_y ? obin_size_y-1 : yend;
		zend   = floor(z_rescaled + ns/2.0)-zoffset;
		zend   = zend >= obin_size_z ? obin_size_z-1 : zend;
#endif
		eval_kernel_vec_Horner(ker1,xstart+xoffset-x_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker2,ystart+yoffset-y_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker3,zstart+zoffset-z_rescaled,ns,sigma);
#if 1
		for(int zz=zstart; zz<=zend; zz++){
			FLT kervalue3 = ker3[zz-zstart];
			for(int yy=ystart; yy<=yend; yy++){
				FLT kervalue2 = ker2[yy-ystart];
				for(int xx=xstart; xx<=xend; xx++){
					outidx = xx+yy*obin_size_x+zz*obin_size_y*obin_size_x;
					FLT kervalue1 = ker1[xx-xstart];
#if 1
					atomicAdd(&fwshared[outidx].x, cnow.x*kervalue1*kervalue2*
						kervalue3);
					atomicAdd(&fwshared[outidx].y, cnow.y*kervalue1*kervalue2*
						kervalue3);
#else
					fwshared[outidx].x+= cnow.x*kervalue1*kervalue2*kervalue3;
					fwshared[outidx].y+= cnow.y*kervalue1*kervalue2*kervalue3;
#endif
				}
			}
		}
#endif
	}
	__syncthreads();
	/* write to global memory */
	for(int n=threadIdx.x; n<N; n+=blockDim.x){
		int i = n%obin_size_x;
		int j = (n/obin_size_x)%obin_size_y;
		int k = n/(obin_size_x*obin_size_y);

		ix = xoffset+i;
		iy = yoffset+j;
		iz = zoffset+k;
		outidx = ix+iy*nf1+iz*nf1*nf2;
		atomicAdd(&fw[outidx].x, fwshared[n].x);
		atomicAdd(&fw[outidx].y, fwshared[n].y);
	}
}

/* ---------------------- 3d Interpolation Kernels ---------------------------*/
/* Kernels for NUptsdriven Method */
__global__
void Interp_3d_NUptsdriven(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta, 
	int *idxnupts, int pirange)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		FLT x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
		FLT y_rescaled=RESCALE(y[idxnupts[i]], nf2, pirange);
		FLT z_rescaled=RESCALE(z[idxnupts[i]], nf3, pirange);
		int xstart = ceil(x_rescaled - ns/2.0);
		int ystart = ceil(y_rescaled - ns/2.0);
		int zstart = ceil(z_rescaled - ns/2.0);
		int xend = floor(x_rescaled + ns/2.0);
		int yend = floor(y_rescaled + ns/2.0);
		int zend = floor(z_rescaled + ns/2.0);
		CUCPX cnow;
		cnow.x = 0.0;
		cnow.y = 0.0;
		for(int zz=zstart; zz<=zend; zz++){
			FLT disz=abs(z_rescaled-zz);
			FLT kervalue3 = evaluate_kernel(disz, es_c, es_beta);
			for(int yy=ystart; yy<=yend; yy++){
				FLT disy=abs(y_rescaled-yy);
				FLT kervalue2 = evaluate_kernel(disy, es_c, es_beta);
				for(int xx=xstart; xx<=xend; xx++){
					int ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
					int iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
					int iz = zz < 0 ? zz+nf3 : (zz>nf3-1 ? zz-nf3 : zz);
					
					int inidx = ix+iy*nf1+iz*nf2*nf1;
					
					FLT disx=abs(x_rescaled-xx);
					FLT kervalue1 = evaluate_kernel(disx, es_c, es_beta);
					cnow.x += fw[inidx].x*kervalue1*kervalue2*kervalue3;
					cnow.y += fw[inidx].y*kervalue1*kervalue2*kervalue3;
				}
			}
		}
		c[idxnupts[i]].x = cnow.x;
		c[idxnupts[i]].y = cnow.y;
	}

}

__global__
void Interp_3d_NUptsdriven_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, 
	int M, const int ns, int nf1, int nf2, int nf3, FLT sigma, int *idxnupts,
	int pirange)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		FLT x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
		FLT y_rescaled=RESCALE(y[idxnupts[i]], nf2, pirange);
		FLT z_rescaled=RESCALE(z[idxnupts[i]], nf3, pirange);

		int xstart = ceil(x_rescaled - ns/2.0);
		int ystart = ceil(y_rescaled - ns/2.0);
		int zstart = ceil(z_rescaled - ns/2.0);

		int xend   = floor(x_rescaled + ns/2.0);
		int yend   = floor(y_rescaled + ns/2.0);
		int zend   = floor(z_rescaled + ns/2.0);

		CUCPX cnow;
		cnow.x = 0.0;
		cnow.y = 0.0;
		
		FLT ker1[MAX_NSPREAD];
		FLT ker2[MAX_NSPREAD];
		FLT ker3[MAX_NSPREAD];

		eval_kernel_vec_Horner(ker1,xstart-x_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker2,ystart-y_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker3,zstart-z_rescaled,ns,sigma);

		for(int zz=zstart; zz<=zend; zz++){
			FLT kervalue3 = ker3[zz-zstart];
			int iz = zz < 0 ? zz+nf3 : (zz>nf3-1 ? zz-nf3 : zz);
			for(int yy=ystart; yy<=yend; yy++){
				FLT kervalue2 = ker2[yy-ystart];
				int iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
				for(int xx=xstart; xx<=xend; xx++){
					int ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
					int inidx = ix+iy*nf1+iz*nf2*nf1;
					FLT kervalue1 = ker1[xx-xstart];
					cnow.x += fw[inidx].x*kervalue1*kervalue2*kervalue3;
					cnow.y += fw[inidx].y*kervalue1*kervalue2*kervalue3;
				}
			}
		}
		c[idxnupts[i]].x = cnow.x;
		c[idxnupts[i]].y = cnow.y;
	}

}

/* Kernels for SubProb Method */
__global__
void Interp_3d_Subprob(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, 
	int M, const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta, 
	int* binstartpts, int* bin_size, int bin_size_x, int bin_size_y, 
	int bin_size_z, int* subprob_to_bin, int* subprobstartpts, int* numsubprob, 
	int maxsubprobsize, int nbinx, int nbiny, int nbinz, int* idxnupts, 
	int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,xend,yend,zstart,zend;
	int subpidx=blockIdx.x;
	int bidx=subprob_to_bin[subpidx];
	int binsubp_idx=subpidx-subprobstartpts[bidx];
	int ix, iy, iz;
	int outidx;
	int ptstart=binstartpts[bidx]+binsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, bin_size[bidx]-binsubp_idx*maxsubprobsize);

	int xoffset=(bidx % nbinx)*bin_size_x;
	int yoffset=((bidx / nbinx)%nbiny)*bin_size_y;
	int zoffset=(bidx/ (nbinx*nbiny))*bin_size_z;

	int N = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0))*
			(bin_size_z+2*ceil(ns/2.0));

#if 1
	for(int n=threadIdx.x;n<N; n+=blockDim.x){
		int i = n % (int) (bin_size_x+2*ceil(ns/2.0) );
		int j = (int) (n /(bin_size_x+2*ceil(ns/2.0))) % (int) (bin_size_y+2*ceil(ns/2.0));
		int k = n / ((bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0)));

		ix = xoffset-ceil(ns/2.0)+i;
		iy = yoffset-ceil(ns/2.0)+j;
		iz = zoffset-ceil(ns/2.0)+k;
		if(ix<(nf1+ceil(ns/2.0)) && iy<(nf2+ceil(ns/2.0)) && iz<(nf3+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			iy = iy < 0 ? iy+nf2 : (iy>nf2-1 ? iy-nf2 : iy);
			iz = iz < 0 ? iz+nf3 : (iz>nf3-1 ? iz-nf3 : iz);
			outidx = ix+iy*nf1+iz*nf1*nf2;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2)+
				k*(bin_size_x+ceil(ns/2.0)*2)*(bin_size_y+ceil(ns/2.0)*2);
			fwshared[sharedidx].x = fw[outidx].x;
			fwshared[sharedidx].y = fw[outidx].y;
		}
	}
#endif
	__syncthreads();

	FLT x_rescaled, y_rescaled, z_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		int idx = ptstart+i;
		x_rescaled=RESCALE(x[idxnupts[idx]], nf1, pirange);
		y_rescaled=RESCALE(y[idxnupts[idx]], nf2, pirange);
		z_rescaled=RESCALE(z[idxnupts[idx]], nf3, pirange);
		cnow.x = 0.0;
		cnow.y = 0.0;

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		zstart = ceil(z_rescaled - ns/2.0)-zoffset;
		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;
		zend   = floor(z_rescaled + ns/2.0)-zoffset;


    	for (int zz=zstart; zz<=zend; zz++){
			FLT disz=abs(z_rescaled-zz);
			FLT kervalue3 = evaluate_kernel(disz, es_c, es_beta);
			iz = zz+ceil(ns/2.0);
			for(int yy=ystart; yy<=yend; yy++){
				FLT disy=abs(y_rescaled-yy);
				FLT kervalue2 = evaluate_kernel(disy, es_c, es_beta);
				iy = yy+ceil(ns/2.0);
				for(int xx=xstart; xx<=xend; xx++){
					ix = xx+ceil(ns/2.0);
					outidx = ix+iy*(bin_size_x+ceil(ns/2.0)*2)+
						iz*(bin_size_x+ceil(ns/2.0)*2)*
						   (bin_size_y+ceil(ns/2.0)*2);
					
					FLT disx=abs(x_rescaled-xx);
					FLT kervalue1 = evaluate_kernel(disx, es_c, es_beta);
					cnow.x += fwshared[outidx].x*kervalue1*kervalue2*kervalue3;
					cnow.y += fwshared[outidx].y*kervalue1*kervalue2*kervalue3;
        		}
      		}
		}
		c[idxnupts[idx]].x = cnow.x;
		c[idxnupts[idx]].y = cnow.y;
	}
}
__global__
void Interp_3d_Subprob_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, 
	int M, const int ns, int nf1, int nf2, int nf3, FLT sigma, int* binstartpts,
	int* bin_size, int bin_size_x, int bin_size_y, int bin_size_z, 
	int* subprob_to_bin, int* subprobstartpts, int* numsubprob, 
	int maxsubprobsize, int nbinx, int nbiny, int nbinz, int* idxnupts,
	int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,xend,yend,zstart,zend;
	int subpidx=blockIdx.x;
	int bidx=subprob_to_bin[subpidx];
	int binsubp_idx=subpidx-subprobstartpts[bidx];
	int ix, iy, iz;
	int outidx;
	int ptstart=binstartpts[bidx]+binsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, bin_size[bidx]-binsubp_idx*maxsubprobsize);

	int xoffset=(bidx % nbinx)*bin_size_x;
	int yoffset=((bidx / nbinx)%nbiny)*bin_size_y;
	int zoffset=(bidx/ (nbinx*nbiny))*bin_size_z;

	int N = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0))*
			(bin_size_z+2*ceil(ns/2.0));

	for(int n=threadIdx.x;n<N; n+=blockDim.x){
		int i = n % (int) (bin_size_x+2*ceil(ns/2.0) );
		int j = (int) (n /(bin_size_x+2*ceil(ns/2.0))) % (int) (bin_size_y+2*ceil(ns/2.0));
		int k = n / ((bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0)));

		ix = xoffset-ceil(ns/2.0)+i;
		iy = yoffset-ceil(ns/2.0)+j;
		iz = zoffset-ceil(ns/2.0)+k;
		if(ix<(nf1+ceil(ns/2.0)) && iy<(nf2+ceil(ns/2.0)) && iz<(nf3+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			iy = iy < 0 ? iy+nf2 : (iy>nf2-1 ? iy-nf2 : iy);
			iz = iz < 0 ? iz+nf3 : (iz>nf3-1 ? iz-nf3 : iz);
			outidx = ix+iy*nf1+iz*nf1*nf2;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2)+
				k*(bin_size_x+ceil(ns/2.0)*2)*(bin_size_y+ceil(ns/2.0)*2);
			fwshared[sharedidx].x = fw[outidx].x;
			fwshared[sharedidx].y = fw[outidx].y;
		}
	}
	__syncthreads();
	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];
	FLT ker3[MAX_NSPREAD];
	FLT x_rescaled, y_rescaled, z_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		int idx = ptstart+i;
		x_rescaled=RESCALE(x[idxnupts[idx]], nf1, pirange);
		y_rescaled=RESCALE(y[idxnupts[idx]], nf2, pirange);
		z_rescaled=RESCALE(z[idxnupts[idx]], nf3, pirange);
		cnow.x = 0.0;
		cnow.y = 0.0;

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		zstart = ceil(z_rescaled - ns/2.0)-zoffset;

		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;
		zend   = floor(z_rescaled + ns/2.0)-zoffset;

		eval_kernel_vec_Horner(ker1,xstart+xoffset-x_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker2,ystart+yoffset-y_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker3,zstart+zoffset-z_rescaled,ns,sigma);
    	for (int zz=zstart; zz<=zend; zz++){
			FLT kervalue3 = ker3[zz-zstart];
			iz = zz+ceil(ns/2.0);
			for(int yy=ystart; yy<=yend; yy++){
				FLT kervalue2 = ker2[yy-ystart];
				iy = yy+ceil(ns/2.0);
				for(int xx=xstart; xx<=xend; xx++){
					ix = xx+ceil(ns/2.0);
					outidx = ix+iy*(bin_size_x+ceil(ns/2.0)*2)+
							 iz*(bin_size_x+ceil(ns/2.0)*2)*
							    (bin_size_y+ceil(ns/2.0)*2);
					FLT kervalue1 = ker1[xx-xstart];
					cnow.x += fwshared[outidx].x*kervalue1*kervalue2*kervalue3;
					cnow.y += fwshared[outidx].y*kervalue1*kervalue2*kervalue3;
        		}
      		}
		}
		c[idxnupts[idx]].x = cnow.x;
		c[idxnupts[idx]].y = cnow.y;
	}
}
#if 0
// This kernels assumes that number of bins less than #maxnumberofthreads in
// each dim
__global__
void CalcSubProb_3d(int bin_size_x, int bin_size_y, int bin_size_z,
	int o_bin_size_x, int o_bin_size_y, int o_bin_size_z, int nbinx, int nbiny,
	int nbinz, int nobinx, int nobiny, int nobinz, int* bin_size,
	int* num_subprob, int* num_nupts, int maxsubprobsize)
{
	int numNUpts = 0;
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
void LocateNUptstoBins(int M, int nf1, int nf2, int nf3, int  bin_size_x,
	int bin_size_y, int bin_size_z, int nbinx, int nbiny, int nbinz,
	int* bin_size, FLT *x, FLT *y, FLT *z, int* sortidx)
{
	int binidx,binx,biny,binz;
	int oldidx;
	FLT x_rescaled,y_rescaled,z_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=RESCALE(x[i], nf1, pirange);
		y_rescaled=RESCALE(y[i], nf2, pirange);
		z_rescaled=RESCALE(z[i], nf3, pirange);
		binx = floor(x_rescaled/bin_size_x);
		biny = floor(y_rescaled/bin_size_y);
		binz = floor(z_rescaled/bin_size_z);
		binidx = binx+biny*nbinx+binz*nbinx*nbiny;
		oldidx = atomicAdd(&bin_size[binidx], 1);
		sortidx[i] = oldidx;
	}
}
#endif

