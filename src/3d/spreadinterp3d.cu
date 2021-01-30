#include <iostream>
#include <math.h>
#include <helper_cuda.h>
#include <cuda.h>
#include "../../contrib/utils.h"
#include "../../contrib/utils_fp.h"
#include "../cuspreadinterp.h"
#include "../precision_independent.h"
#include "../../include/utils.h"

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
			if(iz >= (bin_size_z + (int) ceil(ns/2.0)*2) || iz<0) break;
			for(int yy=ystart; yy<=yend; yy++){
				FLT kervalue2 = ker2[yy-ystart];
				iy = yy+ceil(ns/2.0);
				if(iy >= (bin_size_y + (int) ceil(ns/2.0)*2) || iy<0) break;
				for(int xx=xstart; xx<=xend; xx++){
					ix = xx+ceil(ns/2.0);
					if(ix >= (bin_size_x + (int) ceil(ns/2.0)*2) || ix<0) break;
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
			if(iz >= (bin_size_z + (int) ceil(ns/2.0)*2) || iz<0) break;
			for(int yy=ystart; yy<=yend; yy++){
				FLT kervalue2 = ker2[yy-ystart];
				iy = yy+ceil(ns/2.0);
				if(iy >= (bin_size_y + (int) ceil(ns/2.0)*2) || iy<0) break;
				for(int xx=xstart; xx<=xend; xx++){
					FLT kervalue1 = ker1[xx-xstart];
					ix = xx+ceil(ns/2.0);
					if(ix >= (bin_size_x + (int) ceil(ns/2.0)*2) || ix<0) break;
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
			if(box[d] == 2)
				box[d] = 1;
			b=b/3;
		}
		int ii = idxnupts[idx]%M;
		x_rescaled = RESCALE(x[ii],nf1,pirange) + box[0]*nf1;
		y_rescaled = RESCALE(y[ii],nf2,pirange) + box[1]*nf2;
		z_rescaled = RESCALE(z[ii],nf3,pirange) + box[2]*nf3;
		cnow = c[ii];

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

		for(int zz=zstart; zz<=zend; zz++){
			FLT disz=abs(z_rescaled-(zz+zoffset));
			FLT kervalue3 = evaluate_kernel(disz, es_c, es_beta);
			for(int yy=ystart; yy<=yend; yy++){
				FLT disy=abs(y_rescaled-(yy+yoffset));
				FLT kervalue2 = evaluate_kernel(disy, es_c, es_beta);
				for(int xx=xstart; xx<=xend; xx++){
					outidx = xx+yy*obin_size_x+zz*obin_size_y*obin_size_x;
					FLT disx=abs(x_rescaled-(xx+xoffset));
					FLT kervalue1 = evaluate_kernel(disx, es_c, es_beta);
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
	int xstartnew,ystartnew,zstartnew,xendnew,yendnew,zendnew;
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
			if(box[d] == 2)
				box[d] = 1;
			b=b/3;
		}
		int ii = nidx%M;
		x_rescaled = RESCALE(x[ii],nf1,pirange) + box[0]*nf1;
		y_rescaled = RESCALE(y[ii],nf2,pirange) + box[1]*nf2;
		z_rescaled = RESCALE(z[ii],nf3,pirange) + box[2]*nf3;
		cnow = c[ii];

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		zstart = ceil(z_rescaled - ns/2.0)-zoffset;
		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;
		zend   = floor(z_rescaled + ns/2.0)-zoffset;

		eval_kernel_vec_Horner(ker1,xstart+xoffset-x_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker2,ystart+yoffset-y_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker3,zstart+zoffset-z_rescaled,ns,sigma);

		xstartnew = xstart < 0 ? 0 : xstart;
		ystartnew = ystart < 0 ? 0 : ystart;
		zstartnew = zstart < 0 ? 0 : zstart;
		xendnew   = xend >= obin_size_x ? obin_size_x-1 : xend;
		yendnew   = yend >= obin_size_y ? obin_size_y-1 : yend;
		zendnew   = zend >= obin_size_z ? obin_size_z-1 : zend;

		for(int zz=zstartnew; zz<=zendnew; zz++){
			FLT kervalue3 = ker3[zz-zstart];
			for(int yy=ystartnew; yy<=yendnew; yy++){
				FLT kervalue2 = ker2[yy-ystart];
				for(int xx=xstartnew; xx<=xendnew; xx++){
					outidx = xx+yy*obin_size_x+zz*obin_size_y*obin_size_x;
					FLT kervalue1 = ker1[xx-xstart];
					atomicAdd(&fwshared[outidx].x, cnow.x*kervalue1*kervalue2*kervalue3);
					atomicAdd(&fwshared[outidx].y, cnow.y*kervalue1*kervalue2*kervalue3);
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

