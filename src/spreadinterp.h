
#ifndef __SPREAD_H__
#define __SPREAD_H__

#include "../finufft/utils.h"
#include "cufinufft.h"

#define MAX_NSPREAD 16
//Kernels for 2D codes
__global__
void RescaleXY_2d(int M, int nf1, int nf2, FLT* x, FLT* y);

/* -----------------------------Spreading Kernels-----------------------------*/
/* Kernels for Idriven Method (Scatter) */
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
__global__
void Interp_2d_Idriven(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
	int nf1, int nf2, FLT es_c, FLT es_beta);
__global__
void Interp_2d_Idriven_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, FLT sigma);
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
void Spread_2d_Odriven(int nbin_block_x, int nbin_block_y, int nbinx, int nbiny, int *bin_startpts,
		FLT *x_sorted, FLT *y_sorted, CUCPX *c_sorted, CUCPX *fw, int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta);
__global__
void Spread_2d_Hybrid(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, int* binstartpts,
		int* bin_size, int bin_size_x, int bin_size_y);
__global__
void Spread_2d_Simple(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, int bin_size,
		int bin_size_x, int bin_size_y, int binx, int biny);
#endif


/* CPU wrapper for calling CUDA kernels */
// Wrapper for testing spread, interpolation only
int cufinufft_spread2d(int ms, int mt, int nf1, int nf2, CPX* h_fw, int M, 
	const FLT *h_kx, const FLT *h_ky, const CPX* h_c, FLT eps, 
	cufinufft_plan *d_plan);
int cufinufft_interp2d(int ms, int mt, int nf1, int nf2, CPX* h_fw, int M, 
	FLT *h_kx, FLT *h_ky, CPX* h_c, FLT eps, cufinufft_plan *d_plan);

// Functions for calling different methods of spreading & interpolation
int cuspread2d(cufinufft_plan* d_plan);
int cuinterp2d(cufinufft_plan* d_plan);

// Wrappers for methods of spreading
int cuspread2d_idriven(int nf1, int nf2, int M, cufinufft_plan *d_plan);
int cuspread2d_subprob_prop(int nf1, int nf2, int M, cufinufft_plan *d_plan);
int cuspread2d_paul_prop(int nf1, int nf2, int M, cufinufft_plan *d_plan);
int cuspread2d_subprob(int nf1, int nf2, int M, cufinufft_plan *d_plan);
int cuspread2d_paul(int nf1, int nf2, int M, cufinufft_plan *d_plan);

// Wrappers for methods of interpolation
int cuinterp2d_idriven(int nf1, int nf2, int M, cufinufft_plan *d_plan);
int cuinterp2d_subprob(int nf1, int nf2, int M, cufinufft_plan *d_plan);
#endif
