
#ifndef __SPREAD_H__
#define __SPREAD_H__

#include "../finufft/utils.h"
#include "../finufft/spreadinterp.h"
#include "cufinufft.h"

//Kernels for 2D codes
__global__
void RescaleXY_2d(int M, int nf1, int nf2, FLT* x, FLT* y);

/* -----------------------------Spreading Kernels-----------------------------*/
/* Kernels for Idriven Method */
__global__
void Spread_2d_Idriven(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta);
__global__
void Spread_2d_Idriven_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, FLT sigma);

/* Kernels for SubProb Method */
// SubProb properties
__global__
void CalcBinSize_noghost_2d(int M, int nf1, int nf2, int  bin_size_x, 
	int bin_size_y, int nbinx,int nbiny, int* bin_size, FLT *x, FLT *y, 
	int* sortidx);
__global__
void CalcInvertofGlobalSortIdx_2d(int M, int bin_size_x, int bin_size_y, 
	int nbinx,int nbiny, int* bin_startpts, int* sortidx,FLT *x, FLT *y, 
	int* index);
__global__
void MapBintoSubProb_2d(int* d_subprob_to_bin, int* d_subprobstartpts, 
	int* d_numsubprob,int numbins);
__global__
void CalcSubProb_2d(int* bin_size, int* num_subprob, int maxsubprobsize, 
	int numbins);

// Main Spreading Kernel
__global__
void Spread_2d_Subprob(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, FLT sigma, int* binstartpts,
		int* bin_size, int bin_size_x, int bin_size_y, int* subprob_to_bin,
		int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx, 
		int nbiny,int* idxnupts);
__global__
void Spread_2d_Subprob_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
		const int ns, int nf1, int nf2, FLT sigma, int* binstartpts,
		int* bin_size, int bin_size_x, int bin_size_y, int* subprob_to_bin,
		int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx, 
		int nbiny,int* idxnupts);

/* Kernels for Paul's Method */
__global__
void LocateFineGridPos_Paul(int M, int nf1, int nf2, int  bin_size_x, int bin_size_y, 
		int nbinx, int nbiny, int* bin_size, int ns, FLT *x, FLT *y, 
		int* sortidx, int* finegridsize);
__global__
void CalcInvertofGlobalSortIdx_Paul(int nf1, int nf2, int M, int bin_size_x, 
		int bin_size_y, int nbinx,int nbiny, int ns, FLT *x, FLT *y, 
		int* finegridstartpts, int* sortidx, int* index);
__global__
void CalcSubProb_2d_Paul(int* finegridsize, int* num_subprob, 
	int maxsubprobsize);
__global__
void Spread_2d_Subprob_Paul(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, FLT es_c, FLT es_beta, FLT sigma, 
	int* binstartpts, int* bin_size, int bin_size_x, int bin_size_y, 
	int* subprob_to_bin, int* subprobstartpts, int* numsubprob, 
	int maxsubprobsize, int nbinx, int nbiny, int* idxnupts, int* fgstartpts, 
	int* finegridsize);


/* ---------------------------Interpolation Kernels---------------------------*/
/* Kernels for Idriven Method */
__global__
void Interp_2d_Idriven(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
	int nf1, int nf2, FLT es_c, FLT es_beta);
__global__
void Interp_2d_Idriven_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, FLT sigma);
/* Kernels for Subprob Method */
__global__
void Interp_2d_Subprob(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
	int nf1, int nf2, FLT es_c, FLT es_beta, FLT sigma, int* binstartpts,
	int* bin_size, int bin_size_x, int bin_size_y, int* subprob_to_bin,
	int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx, 
	int nbiny, int* idxnupts);
__global__
void Interp_2d_Subprob_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, FLT sigma, int* binstartpts,
	int* bin_size, int bin_size_x, int bin_size_y, int* subprob_to_bin,
	int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx, 
	int nbiny, int* idxnupts);

#if 0
__global__
void uniformUpdate(int n, int* data, int* buffer);
__global__
void PtsRearrange_noghost_2d(int M, int nf1, int nf2, int bin_size_x, int bin_size_y, int nbinx,
		int nbiny, int* bin_startpts, int* sortidx, FLT *x, FLT *x_sorted,
		FLT *y, FLT *y_sorted, CUCPX *c, CUCPX *c_sorted);
__global__
void Spread_2d_Odriven(int nbin_block_x, int nbin_block_y, int nbinx, int nbiny,
		int *bin_startpts, FLT *x_sorted, FLT *y_sorted, CUCPX *c_sorted,
		CUCPX *fw, int ns, int nf1, int nf2, FLT es_c, FLT es_beta);
__global__
void Spread_2d_Hybrid(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, int* binstartpts,
		int* bin_size, int bin_size_x, int bin_size_y);
__global__
void Spread_2d_Simple(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, int bin_size,
		int bin_size_x, int bin_size_y, int binx, int biny);
#endif

//Kernels for 3D codes
__global__
void RescaleXY_3d(int M, int nf1, int nf2, int nf3, FLT* x, FLT* y, FLT* z);
/* -----------------------------Spreading Kernels-----------------------------*/
/* Kernels for Bin Sort NUpts */
__global__
void CalcBinSize_noghost_3d(int M, int nf1, int nf2, int nf3, int  bin_size_x,
	int bin_size_y, int bin_size_z, int nbinx, int nbiny, int nbinz,
	int* bin_size, FLT *x, FLT *y, FLT *z, int* sortidx);
__global__
void CalcInvertofGlobalSortIdx_3d(int M, int bin_size_x, int bin_size_y,
	int bin_size_z, int nbinx, int nbiny, int nbinz, int* bin_startpts,
	int* sortidx, FLT *x, FLT *y, FLT *z, int* index);
__global__ 
void TrivialGlobalSortIdx_3d(int M, int* index);

/* Kernels for Idriven Method */
__global__
void Spread_3d_Idriven_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT sigma, int* idxnupts);

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
	int maxsubprobsize, int nbinx, int nbiny, int nbinz, int* idxnupts);
__global__
void Spread_3d_Subprob(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta, 
	int* binstartpts,int* bin_size, int bin_size_x, int bin_size_y, 
	int bin_size_z, int* subprob_to_bin, int* subprobstartpts, int* numsubprob, 
	int maxsubprobsize, int nbinx, int nbiny, int nbinz, int* idxnupts);

/* Kernels for Block Gather Method */
__global__
void LocateNUptstoBins_ghost(int M, int  bin_size_x,
	int bin_size_y, int bin_size_z, int nbinx, int nbiny, int nbinz,
	int binsperobinx, int binsperobiny, int binsperobinz, int* bin_size,
	FLT *x, FLT *y, FLT *z, int* sortidx);
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
	int* sortidx, FLT *x, FLT *y, FLT *z, int* index);
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
void Spread_3d_Gather(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta, FLT sigma,
	int* binstartpts, int obin_size_x, int obin_size_y, int obin_size_z,
	int binsperobin, int* subprob_to_bin, int* subprobstartpts,
	int maxsubprobsize, int nobinx, int nobiny, int nobinz, int* idxnupts);
__global__
void Spread_3d_Gather_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta, FLT sigma,
	int* binstartpts, int obin_size_x, int obin_size_y, int obin_size_z,
	int binsperobin, int* subprob_to_bin, int* subprobstartpts,
	int maxsubprobsize, int nobinx, int nobiny, int nobinz, int* idxnupts);

/* -----------------------------Spreading Kernels-----------------------------*/
/* Kernels for Idriven Method */
__global__
void Interp_3d_Idriven_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, 
	int M, const int ns, int nf1, int nf2, int nf3, FLT sigma, int* idxnupts);
__global__
void Interp_3d_Idriven(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta, 
	int* idxnupts);

/* Kernels for Subprob Method */
__global__
void Interp_3d_Subprob_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT sigma, int* binstartpts, 
	int* bin_size, int bin_size_x, int bin_size_y, int bin_size_z, 
	int* subprob_to_bin, int* subprobstartpts, int* numsubprob, 
	int maxsubprobsize, int nbinx, int nbiny, int nbinz, int* idxnupts);

#if 0
__global__
void LocateNUptstoBins(int M, int nf1, int nf2, int nf3, int  bin_size_x,
	int bin_size_y, int bin_size_z, int nbinx, int nbiny, int nbinz,
	int* bin_size, FLT *x, FLT *y, FLT *z, int* sortidx);
__global__
void CalcSubProb_3d(int bin_size_x, int bin_size_y, int bin_size_z,
	int o_bin_size_x, int o_bin_size_y, int o_bin_size_z, int nbinx, int nbiny,
	int nbinz, int nobinx, int nobiny, int nobinz, int* bin_size,
	int* num_subprob, int* num_nupts, int maxsubprobsize);
__global__
void MapBintoSubProb_3d(int* d_subprobstartpts, int* d_subprob_to_bin,
	int* d_subprob_to_nupts, int bin_size_x, int bin_size_y, int bin_size_z,
	int o_bin_size_x, int o_bin_size_y, int o_bin_size_z, int nbinx,
	int nbiny, int nbinz, int nobinx, int nobiny, int nobinz, int* bin_size,
	int* num_subprob, int* num_nupts, int maxsubprobsize);
#endif
/* CPU wrapper for calling CUDA kernels */
// Wrapper for testing spread, interpolation only
int cufinufft_spread2d(int ms, int mt, int nf1, int nf2, CPX* h_fw, int M, 
	const FLT *h_kx, const FLT *h_ky, const CPX* h_c, FLT eps, 
	cufinufft_plan *d_plan);
int cufinufft_interp2d(int ms, int mt, int nf1, int nf2, CPX* h_fw, int M, 
	FLT *h_kx, FLT *h_ky, CPX* h_c, FLT eps, cufinufft_plan *d_plan);
int cufinufft_spread3d(int ms, int mt, int mu, int nf1, int nf2, int nf3,
	CPX* h_fw, int M, const FLT *h_kx, const FLT *h_ky, const FLT* h_z,
	const CPX* h_c, FLT eps, cufinufft_plan *dplan);
int cufinufft_interp3d(int ms, int mt, int mu, int nf1, int nf2, int nf3, 
	CPX* h_fw, int M, FLT *h_kx, FLT *h_ky, FLT *hz, CPX* h_c, FLT eps,
	cufinufft_plan *dplan);

// Functions for calling different methods of spreading & interpolation
int cuspread2d(cufinufft_plan* d_plan);
int cuinterp2d(cufinufft_plan* d_plan);
int cuspread3d(cufinufft_plan* d_plan);
int cuinterp3d(cufinufft_plan* d_plan);

// Wrappers for methods of spreading
int cuspread2d_idriven(int nf1, int nf2, int M, cufinufft_plan *d_plan);
int cuspread2d_subprob_prop(int nf1, int nf2, int M, cufinufft_plan *d_plan);
int cuspread2d_paul_prop(int nf1, int nf2, int M, cufinufft_plan *d_plan);
int cuspread2d_subprob(int nf1, int nf2, int M, cufinufft_plan *d_plan);
int cuspread2d_paul(int nf1, int nf2, int M, cufinufft_plan *d_plan);

int cuspread3d_idriven_prop(int nf1, int nf2, int nf3, int M,
	cufinufft_plan *d_plan);
int cuspread3d_idriven(int nf1, int nf2, int nf3, int M, cufinufft_plan *d_plan);
int cuspread3d_gather_prop(int nf1, int nf2, int nf3, int M,
	cufinufft_plan *d_plan);
int cuspread3d_gather(int nf1, int nf2, int nf3, int M, cufinufft_plan *d_plan);
int cuspread3d_subprob_prop(int nf1, int nf2, int nf3, int M,
	cufinufft_plan *d_plan);
int cuspread3d_subprob(int nf1, int nf2, int nf3, int M, cufinufft_plan *d_plan);

// Wrappers for methods of interpolation
int cuinterp2d_idriven(int nf1, int nf2, int M, cufinufft_plan *d_plan);
int cuinterp2d_subprob(int nf1, int nf2, int M, cufinufft_plan *d_plan);
int cuinterp3d_idriven(int nf1, int nf2, int nf3, int M, cufinufft_plan *d_plan);
int cuinterp3d_subprob(int nf1, int nf2, int nf3, int M, cufinufft_plan *d_plan);
#endif
