#ifndef __CUSPREADINTERP_H__
#define __CUSPREADINTERP_H__

#include <cufinufft_eitherprec.h>

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
int CUSPREAD2D(CUFINUFFT_PLAN d_plan, int blksize);
int CUINTERP2D(CUFINUFFT_PLAN d_plan, int blksize);
int CUSPREAD3D(CUFINUFFT_PLAN d_plan, int blksize);
int CUINTERP3D(CUFINUFFT_PLAN d_plan, int blksize);

// Wrappers for methods of spreading
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
int CUINTERP2D_NUPTSDRIVEN(int nf1, int nf2, int M, CUFINUFFT_PLAN d_plan,
	int blksize);
int CUINTERP2D_SUBPROB(int nf1, int nf2, int M, CUFINUFFT_PLAN d_plan,
	int blksize);
int CUINTERP3D_NUPTSDRIVEN(int nf1, int nf2, int nf3, int M,
	CUFINUFFT_PLAN d_plan, int blksize);
int CUINTERP3D_SUBPROB(int nf1, int nf2, int nf3, int M, CUFINUFFT_PLAN d_plan,
	int blksize);
#endif
