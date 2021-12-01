#ifndef __CUSPREADINTERP_H__
#define __CUSPREADINTERP_H__

#include <cufinufft_eitherprec.h>

static __forceinline__ __device__
FLT evaluate_kernel(FLT x, FLT es_c, FLT es_beta, int ns)
/* ES ("exp sqrt") kernel evaluation at single real argument:
   phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2
   related to an asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   This is the "reference implementation", used by eg common/onedim_* 
    2/17/17 */
{
	return abs(x) < ns/2.0 ? exp(es_beta * (sqrt(1.0 - es_c*x*x))) : 0.0;
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
#include "../contrib/ker_horner_allw_loop.c"
	}
}

static __inline__ __device__
void eval_kernel_vec(FLT *ker, const FLT x, const double w, const double es_c, 
					 const double es_beta)
{
	for(int i=0; i<w; i++){
		ker[i] = evaluate_kernel(abs(x+i), es_c, es_beta, w);		
	}
}

//Kernels for 1D codes
/* -----------------------------Spreading Kernels-----------------------------*/
/* Kernels for NUptsdriven Method */
__global__
void Spread_1d_NUptsdriven(FLT *x, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, FLT es_c, FLT es_beta, int* idxnupts, int pirange);
__global__
void Spread_1d_NUptsdriven_Horner(FLT *x, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, FLT sigma, int* idxnupts, int pirange);

/* Kernels for SubProb Method */
// SubProb properties
__global__
void CalcBinSize_noghost_1d(int M, int nf1, int  bin_size_x,
	int nbinx, int* bin_size, FLT *x, int* sortidx, int pirange);
__global__
void CalcInvertofGlobalSortIdx_1d(int M, int bin_size_x, int nbinx, 
	int* bin_startpts, int* sortidx,FLT *x, int* index, int pirange, int nf1);

// Main Spreading Kernel
__global__
void Spread_1d_Subprob(FLT *x, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, FLT es_c, FLT es_beta, FLT sigma, int* binstartpts,
		int* bin_size, int bin_size_x, int* subprob_to_bin,
		int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx,
		int* idxnupts, int pirange);
__global__
void Spread_1d_Subprob_Horner(FLT *x, CUCPX *c, CUCPX *fw, int M,
		const int ns, int nf1, FLT sigma, int* binstartpts,
		int* bin_size, int bin_size_x, int* subprob_to_bin,
		int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx,
		int* idxnupts, int pirange);
/* ---------------------------Interpolation Kernels---------------------------*/
/* Kernels for NUptsdriven Method */
__global__
void Interp_1d_NUptsdriven(FLT *x, CUCPX *c, CUCPX *fw, int M, const int ns,
	int nf2, FLT es_c, FLT es_beta, int *idxnupts, int pirange);
__global__
void Interp_1d_NUptsdriven_Horner(FLT *x, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf2, FLT sigma, int *idxnupts, int pirange);

//Kernels for 2D codes
/* -----------------------------Spreading Kernels-----------------------------*/
/* Kernels for NUptsdriven Method */
__global__
void Spread_2d_NUptsdriven(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, int* idxnupts, int pirange);
__global__
void Spread_2d_NUptsdriven_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, FLT sigma, int* idxnupts, int pirange);

/* Kernels for SubProb Method */
// SubProb properties
__global__
void CalcBinSize_noghost_2d(int M, int nf1, int nf2, int  bin_size_x,
	int bin_size_y, int nbinx,int nbiny, int* bin_size, FLT *x, FLT *y,
	int* sortidx, int pirange);
__global__
void CalcInvertofGlobalSortIdx_2d(int M, int bin_size_x, int bin_size_y,
	int nbinx,int nbiny, int* bin_startpts, int* sortidx,FLT *x, FLT *y,
	int* index, int pirange, int nf1, int nf2);

// Main Spreading Kernel
__global__
void Spread_2d_Subprob(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, FLT sigma, int* binstartpts,
		int* bin_size, int bin_size_x, int bin_size_y, int* subprob_to_bin,
		int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx,
		int nbiny,int* idxnupts, int pirange);
__global__
void Spread_2d_Subprob_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M,
		const int ns, int nf1, int nf2, FLT sigma, int* binstartpts,
		int* bin_size, int bin_size_x, int bin_size_y, int* subprob_to_bin,
		int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx,
		int nbiny,int* idxnupts, int pirange);

/* Kernels for Paul's Method */
__global__
void LocateFineGridPos_Paul(int M, int nf1, int nf2, int  bin_size_x, int bin_size_y,
		int nbinx, int nbiny, int* bin_size, int ns, FLT *x, FLT *y,
		int* sortidx, int* finegridsize, int pirange);
__global__
void CalcInvertofGlobalSortIdx_Paul(int nf1, int nf2, int M, int bin_size_x,
	int bin_size_y, int nbinx,int nbiny, int ns, FLT *x, FLT *y,
	int* finegridstartpts, int* sortidx, int* index, int pirange);
__global__
void Spread_2d_Subprob_Paul(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, FLT es_c, FLT es_beta, FLT sigma,
	int* binstartpts, int* bin_size, int bin_size_x, int bin_size_y,
	int* subprob_to_bin, int* subprobstartpts, int* numsubprob,
	int maxsubprobsize, int nbinx, int nbiny, int* idxnupts, int* fgstartpts,
	int* finegridsize, int pirange);


/* ---------------------------Interpolation Kernels---------------------------*/
/* Kernels for NUptsdriven Method */
__global__
void Interp_2d_NUptsdriven(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
	int nf1, int nf2, FLT es_c, FLT es_beta, int *idxnupts, int pirange);
__global__
void Interp_2d_NUptsdriven_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, FLT sigma, int *idxnupts, int pirange);
/* Kernels for Subprob Method */
__global__
void Interp_2d_Subprob(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
	int nf1, int nf2, FLT es_c, FLT es_beta, FLT sigma, int* binstartpts,
	int* bin_size, int bin_size_x, int bin_size_y, int* subprob_to_bin,
	int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx,
	int nbiny, int* idxnupts, int pirange);
__global__
void Interp_2d_Subprob_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, FLT sigma, int* binstartpts,
	int* bin_size, int bin_size_x, int bin_size_y, int* subprob_to_bin,
	int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx,
	int nbiny, int* idxnupts, int pirange);

//Kernels for 3D codes
/* -----------------------------Spreading Kernels-----------------------------*/
/* Kernels for Bin Sort NUpts */
__global__
void CalcBinSize_noghost_3d(int M, int nf1, int nf2, int nf3, int  bin_size_x,
	int bin_size_y, int bin_size_z, int nbinx, int nbiny, int nbinz,
	int* bin_size, FLT *x, FLT *y, FLT *z, int* sortidx, int pirange);
__global__
void CalcInvertofGlobalSortIdx_3d(int M, int bin_size_x, int bin_size_y,
	int bin_size_z, int nbinx, int nbiny, int nbinz, int* bin_startpts,
	int* sortidx, FLT *x, FLT *y, FLT *z, int* index, int pirange, int nf1,
	int nf2, int nf3);
__global__
void TrivialGlobalSortIdx_3d(int M, int* index);

/* Kernels for NUptsdriven Method */
__global__
void Spread_3d_NUptsdriven_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw,
	int M, const int ns, int nf1, int nf2, int nf3, FLT sigma, int* idxnupts,
	int pirange);
__global__
void Spread_3d_NUptsdriven(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta,
	int* idxnupts, int pirange);

/* Kernels for Subprob Method */
__global__
void CalcSubProb_3d_v2(int* bin_size, int* num_subprob, int maxsubprobsize,
	int numbins);
__global__
void MapBintoSubProb_3d_v2(int* d_subprob_to_bin,int* d_subprobstartpts,
	int* d_numsubprob,int numbins);
__global__
void Spread_3d_Subprob_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw,
	int M, const int ns, int nf1, int nf2, int nf3, FLT sigma, int* binstartpts,
	int* bin_size, int bin_size_x, int bin_size_y, int bin_size_z,
	int* subprob_to_bin, int* subprobstartpts, int* numsubprob,
	int maxsubprobsize, int nbinx, int nbiny, int nbinz, int* idxnupts,
	int pirange);
__global__
void Spread_3d_Subprob(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta,
	int* binstartpts,int* bin_size, int bin_size_x, int bin_size_y,
	int bin_size_z, int* subprob_to_bin, int* subprobstartpts, int* numsubprob,
	int maxsubprobsize, int nbinx, int nbiny, int nbinz, int* idxnupts,
	int pirange);

/* Kernels for Block BlockGather Method */
__global__
void LocateNUptstoBins_ghost(int M, int  bin_size_x,
	int bin_size_y, int bin_size_z, int nbinx, int nbiny, int nbinz,
	int binsperobinx, int binsperobiny, int binsperobinz, int* bin_size,
	FLT *x, FLT *y, FLT *z, int* sortidx, int pirange, int nf1, int nf2,
	int nf3);
__global__
void Temp(int binsperobinx, int binsperobiny, int binsperobinz,
	int nbinx, int nbiny, int nbinz, int* binsize);
__global__
void FillGhostBins(int binsperobinx, int binsperobiny, int binsperobinz,
	int nbinx, int nbiny, int nbinz, int* binsize);
__global__
void CalcInvertofGlobalSortIdx_ghost(int M, int  bin_size_x,
	int bin_size_y, int bin_size_z, int nbinx, int nbiny, int nbinz,
	int binsperobinx, int binsperobiny, int binsperobinz, int* bin_startpts,
	int* sortidx, FLT *x, FLT *y, FLT *z, int* index, int pirange,
	int nf1, int nf2, int nf3);
__global__
void GhostBinPtsIdx(int binsperobinx, int binsperobiny, int binsperobinz,
	int nbinx, int nbiny, int nbinz, int* binsize, int* index,
	int* bin_startpts, int M);
__global__
void CalcSubProb_3d_v1(int binsperobinx, int binsperobiny, int binsperobinz,
	int* bin_size, int* num_subprob, int maxsubprobsize, int numbins);
__global__
void MapBintoSubProb_3d_v1(int* d_subprob_to_obin, int* d_subprobstartpts,
	int* d_numsubprob,int numbins);
__global__
void Spread_3d_BlockGather(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta, FLT sigma,
	int* binstartpts, int obin_size_x, int obin_size_y, int obin_size_z,
	int binsperobin, int* subprob_to_bin, int* subprobstartpts,
	int maxsubprobsize, int nobinx, int nobiny, int nobinz, int* idxnupts,
	int pirange);
__global__
void Spread_3d_BlockGather_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta, FLT sigma,
	int* binstartpts, int obin_size_x, int obin_size_y, int obin_size_z,
	int binsperobin, int* subprob_to_bin, int* subprobstartpts,
	int maxsubprobsize, int nobinx, int nobiny, int nobinz, int* idxnupts,
	int pirange);

/* -----------------------------Spreading Kernels-----------------------------*/
/* Kernels for NUptsdriven Method */
__global__
void Interp_3d_NUptsdriven_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw,
	int M, const int ns, int nf1, int nf2, int nf3, FLT sigma, int* idxnupts,
	int pirange);
__global__
void Interp_3d_NUptsdriven(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta,
	int* idxnupts, int pirange);

/* Kernels for Subprob Method */
__global__
void Interp_3d_Subprob_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT sigma, int* binstartpts,
	int* bin_size, int bin_size_x, int bin_size_y, int bin_size_z,
	int* subprob_to_bin, int* subprobstartpts, int* numsubprob,
	int maxsubprobsize, int nbinx, int nbiny, int nbinz, int* idxnupts,
	int pirange);
__global__
void Interp_3d_Subprob(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw,
	int M, const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta,
	int* binstartpts, int* bin_size, int bin_size_x, int bin_size_y,
	int bin_size_z, int* subprob_to_bin, int* subprobstartpts, int* numsubprob,
	int maxsubprobsize, int nbinx, int nbiny, int nbinz, int* idxnupts,
	int pirange);

/* C wrapper for calling CUDA kernels */
// Wrapper for testing spread, interpolation only
int CUFINUFFT_SPREAD1D(int nf1, CUCPX* d_fw, int M,
	FLT *d_kx, CUCPX* d_c, CUFINUFFT_PLAN d_plan);
int CUFINUFFT_INTERP1D(int nf1, CUCPX* d_fw, int M,
	FLT *d_kx, CUCPX* d_c, CUFINUFFT_PLAN d_plan);
int CUFINUFFT_SPREAD2D(int nf1, int nf2, CUCPX* d_fw, int M,
	FLT *d_kx, FLT *d_ky, CUCPX* d_c, CUFINUFFT_PLAN d_plan);
int CUFINUFFT_INTERP2D(int nf1, int nf2, CUCPX* d_fw, int M,
	FLT *d_kx, FLT *d_ky, CUCPX* d_c, CUFINUFFT_PLAN d_plan);
int CUFINUFFT_SPREAD3D(int nf1, int nf2, int nf3,
	CUCPX* d_fw, int M, FLT *d_kx, FLT *d_ky, FLT* d_kz,
	CUCPX* d_c, CUFINUFFT_PLAN dplan);
int CUFINUFFT_INTERP3D(int nf1, int nf2, int nf3,
	CUCPX* d_fw, int M, FLT *d_kx, FLT *d_ky, FLT *d_kz, 
    CUCPX* d_c, CUFINUFFT_PLAN dplan);

// Functions for calling different methods of spreading & interpolation
int CUSPREAD1D(CUFINUFFT_PLAN d_plan, int blksize);
int CUINTERP1D(CUFINUFFT_PLAN d_plan, int blksize);
int CUSPREAD2D(CUFINUFFT_PLAN d_plan, int blksize);
int CUINTERP2D(CUFINUFFT_PLAN d_plan, int blksize);
int CUSPREAD3D(CUFINUFFT_PLAN d_plan, int blksize);
int CUINTERP3D(CUFINUFFT_PLAN d_plan, int blksize);

// Wrappers for methods of spreading
int CUSPREAD1D_NUPTSDRIVEN_PROP(int nf1, int M, CUFINUFFT_PLAN d_plan);
int CUSPREAD1D_NUPTSDRIVEN(int nf1, int M, CUFINUFFT_PLAN d_plan, int blksize);
int CUSPREAD1D_SUBPROB_PROP(int nf1, int M, CUFINUFFT_PLAN d_plan);
int CUSPREAD1D_SUBPROB(int nf1, int M, CUFINUFFT_PLAN d_plan, int blksize);

int CUSPREAD2D_NUPTSDRIVEN_PROP(int nf1, int nf2, int M, CUFINUFFT_PLAN d_plan);
int CUSPREAD2D_NUPTSDRIVEN(int nf1, int nf2, int M, CUFINUFFT_PLAN d_plan,
	int blksize);
int CUSPREAD2D_SUBPROB_PROP(int nf1, int nf2, int M, CUFINUFFT_PLAN d_plan);
int CUSPREAD2D_PAUL_PROP(int nf1, int nf2, int M, CUFINUFFT_PLAN d_plan);
int CUSPREAD2D_SUBPROB(int nf1, int nf2, int M, CUFINUFFT_PLAN d_plan,
	int blksize);
int CUSPREAD2D_PAUL(int nf1, int nf2, int M, CUFINUFFT_PLAN d_plan,
	int blksize);

int CUSPREAD3D_NUPTSDRIVEN_PROP(int nf1, int nf2, int nf3, int M,
	CUFINUFFT_PLAN d_plan);
int CUSPREAD3D_NUPTSDRIVEN(int nf1, int nf2, int nf3, int M,
	CUFINUFFT_PLAN d_plan, int blksize);
int CUSPREAD3D_BLOCKGATHER_PROP(int nf1, int nf2, int nf3, int M,
	CUFINUFFT_PLAN d_plan);
int CUSPREAD3D_BLOCKGATHER(int nf1, int nf2, int nf3, int M,
	CUFINUFFT_PLAN d_plan, int blksize);
int CUSPREAD3D_SUBPROB_PROP(int nf1, int nf2, int nf3, int M,
	CUFINUFFT_PLAN d_plan);
int CUSPREAD3D_SUBPROB(int nf1, int nf2, int nf3, int M, CUFINUFFT_PLAN d_plan,
	int blksize);

// Wrappers for methods of interpolation
int CUINTERP1D_NUPTSDRIVEN(int nf1, int M, CUFINUFFT_PLAN d_plan, int blksize);
int CUINTERP2D_NUPTSDRIVEN(int nf1, int nf2, int M, CUFINUFFT_PLAN d_plan,
	int blksize);
int CUINTERP2D_SUBPROB(int nf1, int nf2, int M, CUFINUFFT_PLAN d_plan,
	int blksize);
int CUINTERP3D_NUPTSDRIVEN(int nf1, int nf2, int nf3, int M,
	CUFINUFFT_PLAN d_plan, int blksize);
int CUINTERP3D_SUBPROB(int nf1, int nf2, int nf3, int M, CUFINUFFT_PLAN d_plan,
	int blksize);
#endif
