#ifndef __SPREAD_H__
#define __SPREAD_H__

#include "../finufft/utils.h"
#include "cufinufft.h"

#define MAX_NSPREAD 16
//Kernels for 1D codes (this is outdated ... )
/*
   __global__
   void CalcBinSize_1d(int M, int nf1, int  bin_size_x, int nbinx,
   int* bin_size, FLT *x, int* sortidx);
   __global__
   void FillGhostBin_1d(int bin_size_x, int nbinx, int*bin_size);

   __global__
   void BinsStartPts_1d(int M, int totalnumbins, int* bin_size, int* bin_startpts);

   __global__
   void PtsRearrage_1d(int M, int nf1, int bin_size_x, int nbinx,
   int* bin_startpts, int* sortidx, FLT *x, FLT *x_sorted,
   FLT *c, FLT *c_sorted);
   __global__
   void Spread_1d(int nbin_block_x, int nbinx, int *bin_startpts,
   FLT *x_sorted, FLT *c_sorted, FLT *fw, int ns,
   int nf1, FLT es_c, FLT es_beta);
   */

#if 0
__global__
void RescaleXY_1d(int M, int nf1, FLT* x);
//Kernels for 1D codes
__global__
void CalcBinSize_noghost_1d(int M, int nf1, int nf2, int  bin_size_x, int bin_size_y, int nbinx,
		int nbiny, int* bin_size, FLT *x, FLT *y, int* sortidx);
__global__
void uniformUpdate(int n, int* data, int* buffer);
__global__
void CalcInvertofGlobalSortIdx_1d(int M, int bin_size_x, int bin_size_y, int nbinx,
		int nbiny, int* bin_startpts, int* sortidx,
		FLT *x, FLT *y, int* index);
__global__
void PtsRearrage_noghost_1d(int M, int nf1, int nf2, int bin_size_x, int bin_size_y, int nbinx,
		int nbiny, int* bin_startpts, int* sortidx, FLT *x, FLT *x_sorted,
		FLT *y, FLT *y_sorted, CUCPX *c, CUCPX *c_sorted);
__global__
void Spread_1d_Odriven(int nbin_block_x, int nbin_block_y, int nbinx, int nbiny, int *bin_startpts,
		FLT *x_sorted, FLT *y_sorted, CUCPX *c_sorted, CUCPX *fw, int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, int fw_width);
__global__
void Spread_1d_Idriven(FLT *x, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, FLT es_c, FLT es_beta, int fw_width);
__global__
void Interp_1d_Idriven(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, int fw_width);
__global__
void Spread_1d_Idriven_Horner(FLT *x, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, FLT es_c, FLT es_beta, int fw_width);
__global__
void Spread_1d_Hybrid(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, int fw_width, int* binstartpts,
		int* bin_size, int bin_size_x, int bin_size_y);
__global__
void Spread_1d_Simple(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, int fw_width, int bin_size,
		int bin_size_x, int bin_size_y, int binx, int biny);
__global__
void CalcSubProb_1d(int* bin_size, int* num_subprob, int maxsubprobsize, int numbins);
__global__
void MapBintoSubProb_1d(int* d_subprob_to_bin, int* d_subprobstartpts, int* d_numsubprob,
		int numbins);
__global__
void Spread_1d_Subprob(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, FLT sigma, int fw_width, int* binstartpts,
		int* bin_size, int bin_size_x, int bin_size_y, int* subprob_to_bin,
		int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx, int nbiny,
		int* idxnupts);
__global__
void Interp_1d_Subprob(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, FLT sigma, int fw_width, int* binstartpts,
		int* bin_size, int bin_size_x, int bin_size_y, int* subprob_to_bin,
		int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx, int nbiny,
		int* idxnupts);
#endif

//Kernels for 2D codes
__global__
void RescaleXY_2d(int M, int nf1, int nf2, FLT* x, FLT* y);
__global__
void CalcBinSize_noghost_2d(int M, int nf1, int nf2, int  bin_size_x, 
	int bin_size_y, int nbinx, int nbiny, int* bin_size, FLT *x, FLT *y, 
	int* sortidx);
__global__
void uniformUpdate(int n, int* data, int* buffer);
__global__
void CalcInvertofGlobalSortIdx_2d(int M, int bin_size_x, int bin_size_y, 
		int nbinx, int nbiny, int* bin_startpts, int* sortidx, FLT *x, FLT *y, 
		int* index);
__global__
void PtsRearrage_noghost_2d(int M, int nf1, int nf2, int bin_size_x, 
		int bin_size_y, int nbinx, int nbiny, int* bin_startpts, 
		int* sortidx, FLT *x, FLT *x_sorted, FLT *y, FLT *y_sorted, CUCPX *c, 
		CUCPX *c_sorted);
__global__
void Spread_2d_Odriven(int nbin_block_x, int nbin_block_y, int nbinx, int nbiny, 
		int *bin_startpts, FLT *x_sorted, FLT *y_sorted, CUCPX *c_sorted, 
		CUCPX *fw, int ns, int nf1, int nf2, FLT es_c, FLT es_beta);
__global__
void Spread_2d_Idriven(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta);
__global__
void Spread_2d_Idriven_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
		const int ns, int nf1, int nf2, FLT es_c, FLT es_beta);
__global__
void Interp_2d_Idriven(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta);
__global__
void Interp_2d_Idriven_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
		const int ns, int nf1, int nf2, FLT sigma);
__global__
void Spread_2d_Hybrid(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, int* binstartpts,
		int* bin_size, int bin_size_x, int bin_size_y);
__global__
void Spread_2d_Simple(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, int bin_size,
		int bin_size_x, int bin_size_y, int binx, int biny);
__global__
void CalcSubProb_2d(int* bin_size, int* num_subprob, int maxsubprobsize, 
		int numbins);

__global__
void CalcSubProb_2d_Paul(int* finegridsize, int* num_subprob, 
	int maxsubprobsize);

__global__
void MapBintoSubProb_2d(int* d_subprob_to_bin, int* d_subprobstartpts, 
		int* d_numsubprob, int numbins);
__global__
void Spread_2d_Subprob(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, FLT sigma, int* binstartpts,
		int* bin_size, int bin_size_x, int bin_size_y, int* subprob_to_bin,
		int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx, 
		int nbiny, int* idxnupts);
__global__
void Spread_2d_Subprob_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
		const int ns, int nf1, int nf2, FLT sigma, int* binstartpts,
		int* bin_size, int bin_size_x, int bin_size_y, int* subprob_to_bin,
		int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx, 
		int nbiny, int* idxnupts);
__global__
void Spread_2d_Subprob_Horner_Paul(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, FLT es_c, FLT es_beta, FLT sigma, 
	int* binstartpts, int* bin_size, int bin_size_x, int bin_size_y, 
	int* subprob_to_bin, int* subprobstartpts, int* numsubprob, 
	int maxsubprobsize, int nbinx, int nbiny, int* idxnupts, int* fgstartpts, 
	int* finegridsize);
__global__
void Interp_2d_Subprob(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		int nf1, int nf2, FLT es_c, FLT es_beta, FLT sigma, int* binstartpts,
		int* bin_size, int bin_size_x, int bin_size_y, int* subprob_to_bin,
		int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx, int nbiny,
		int* idxnupts);
__global__
void Interp_2d_Subprob_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
		const int ns, int nf1, int nf2, FLT sigma, int* binstartpts,
		int* bin_size, int bin_size_x, int bin_size_y, int* subprob_to_bin,
		int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx, 
		int nbiny, int* idxnupts);

//Kernels for 3D codes
__global__
void RescaleXY_3d(int M, int nf1, int nf2, int nf3, FLT* x, FLT* y, FLT* z);
__global__
void LocateNUptstoBins(int M, int nf1, int nf2, int nf3, int  bin_size_x, 
	int bin_size_y, int bin_size_z, int nbinx, int nbiny, int nbinz, 
	int* bin_size, FLT *x, FLT *y, FLT *z, int* sortidx);
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
void GhostBinPtsIdx(int binsperobinx, int binsperobiny, int binsperobinz, 
	int nbinx, int nbiny, int nbinz, int* binsize, int* index, 
	int* bin_startpts, int M);
__global__
void CalcSubProb_3d_v1(int binsperobinx, int binsperobiny, int binsperobinz, 
	int* bin_size, int* num_subprob, int maxsubprobsize, int numbins);
__global__
void CalcSubProb_3d(int bin_size_x, int bin_size_y, int bin_size_z, 
	int o_bin_size_x, int o_bin_size_y, int o_bin_size_z, int nbinx, int nbiny, 
	int nbinz, int nobinx, int nobiny, int nobinz, int* bin_size, 
	int* num_subprob, int* num_nupts, int maxsubprobsize);
__global__
void CalcInvertofGlobalSortIdx_3d(int M, int bin_size_x, int bin_size_y, 
	int bin_size_z, int nbinx, int nbiny, int nbinz, int* bin_startpts, 
	int* sortidx, FLT *x, FLT *y, FLT *z, int* index);
__global__
void CalcInvertofGlobalSortIdx_ghost(int M, int  bin_size_x, 
	int bin_size_y, int bin_size_z, int nbinx, int nbiny, int nbinz, 
	int binsperobinx, int binsperobiny, int binsperobinz, int* bin_startpts, 
	int* sortidx, FLT *x, FLT *y, FLT *z, int* index);
__global__
void Spread_3d_Idriven_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, FLT es_c, FLT es_beta);
__global__
void CalcSubProb_3d(int* bin_size, int* num_subprob, int maxsubprobsize, 
	int numbins);
__global__
void MapBintoSubProb_3d_v1(int* d_subprob_to_obin, int* d_subprobstartpts, 
	int* d_numsubprob,int numbins);
__global__
void MapBintoSubProb_3d(int* d_subprobstartpts, int* d_subprob_to_bin, 
	int* d_subprob_to_nupts, int bin_size_x, int bin_size_y, int bin_size_z, 
	int o_bin_size_x, int o_bin_size_y, int o_bin_size_z, int nbinx, 
	int nbiny, int nbinz, int nobinx, int nobiny, int nobinz, int* bin_size, 
	int* num_subprob, int* num_nupts, int maxsubprobsize);
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
__global__
void Spread_3d_Subprob_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, FLT sigma, int* binstartpts, int* bin_size, 
	int bin_size_x, int bin_size_y, int* subprob_to_bin, int* subprobstartpts, 
	int* numsubprob, int maxsubprobsize, int nbinx, int nbiny, int* idxnupts);
// Paul
__global__
void LocateFineGridPos(int M, int nf1, int nf2, int  bin_size_x, int bin_size_y, 
		int nbinx, int nbiny, int* bin_size, int ns, FLT *x, FLT *y, 
		int* sortidx, int* finegridsize);
__global__
void CalcInvertofGlobalSortIdx_Paul(int nf1, int nf2, int M, int bin_size_x, 
		int bin_size_y, int nbinx,int nbiny, int ns, FLT *x, FLT *y, 
		int* finegridstartpts, int* sortidx, int* index);
#if 0
// 1d
int cufinufft_spread1d(int ms, int nf1, CPX* h_fw, int M, FLT *h_kx,
		CPX* h_c, cufinufft_opts &opts, cufinufft_plan *dmem);
int cufinufft_interp1d(int ms, int mt, int nf1, int nf2, CPX* h_fw, int M, FLT *h_kx,
		FLT *h_ky, CPX* h_c, cufinufft_opts &opts, cufinufft_plan *dmem);
int cuspread1d_idriven(int nf1, int fw_width, int M, const cufinufft_opts opts,
		cufinufft_plan *d_mem);
int cuinterp1d_idriven(int nf1, int nf2, int fw_width, int M, const cufinufft_opts opts,
		cufinufft_plan *d_mem);
int cuspread1d_idriven_sorted(int nf1, int nf2, int fw_width, int M, const cufinufft_opts opts,
		cufinufft_plan *d_mem);
int cuspread1d_hybrid(int nf1, int nf2, int fw_width, int M, const cufinufft_opts opts,
		cufinufft_plan *d_mem);
int cuspread1d_subprob(int nf1, int nf2, int fw_width, int M, const cufinufft_opts opts,
		cufinufft_plan *d_mem);
int cuinterp1d_subprob(int nf1, int nf2, int fw_width, int M, const cufinufft_opts opts,
		cufinufft_plan *d_mem);
int cuspread1d_simple(int nf1, int nf2, int fw_width, CUCPX* d_fw, int M, FLT *d_kx,
		FLT *d_ky, CUCPX *d_c, const cufinufft_opts opts, int binx, int biny);
int cuspread1d(cufinufft_opts &opts, cufinufft_plan* d_plan);
int cuinterp1d(cufinufft_opts &opts, cufinufft_plan* d_plan);
#endif

// 2d
int cufinufft_spread2d(int ms, int mt, int nf1, int nf2, CPX* h_fw, int M, 
		const FLT *h_kx, const FLT *h_ky, const CPX* h_c, cufinufft_opts &opts, 
		cufinufft_plan *dmem);
int cufinufft_interp2d(int ms, int mt, int nf1, int nf2, CPX* h_fw, int M, 
	FLT *h_kx, FLT *h_ky, CPX* h_c, cufinufft_opts &opts, cufinufft_plan *dmem);
int cuspread2d_idriven(int nf1, int nf2, int M, const cufinufft_opts opts,
		cufinufft_plan *d_mem);
int cuinterp2d_idriven(int nf1, int nf2, int M, const cufinufft_opts opts,
		cufinufft_plan *d_mem);
int cuspread2d_idriven_sorted(int nf1, int nf2, int M, const cufinufft_opts opts,
		cufinufft_plan *d_mem);
int cuspread2d_hybrid(int nf1, int nf2, int M, const cufinufft_opts opts,
		cufinufft_plan *d_mem);
int cuspread2d_subprob_prop(int nf1, int nf2, int M, const cufinufft_opts opts, 
		cufinufft_plan *d_plan);
int cuspread2d_paul_prop(int nf1, int nf2, int M, const cufinufft_opts opts, 
		cufinufft_plan *d_plan);
int cuspread2d_subprob(int nf1, int nf2, int M, const cufinufft_opts opts,
		cufinufft_plan *d_mem);
int cuspread2d_paul(int nf1, int nf2, int M, const cufinufft_opts opts,
		cufinufft_plan *d_mem);
int cuinterp2d_subprob(int nf1, int nf2, int M, const cufinufft_opts opts,
		cufinufft_plan *d_mem);
int cuspread2d_simple(int nf1, int nf2, CUCPX* d_fw, int M, FLT *d_kx,
		FLT *d_ky, CUCPX *d_c, const cufinufft_opts opts, int binx, int biny);
int cuspread2d(cufinufft_opts &opts, cufinufft_plan* d_plan);
int cuinterp2d(cufinufft_opts &opts, cufinufft_plan* d_plan);

// 3d
int cufinufft_spread3d(int ms, int mt, int mu, int nf1, int nf2, int nf3, 
		CPX* h_fw, int M, const FLT *h_kx, const FLT *h_ky, const FLT* h_z, 
		const CPX* h_c, cufinufft_opts &opts, cufinufft_plan *dplan);
int cuspread3d(cufinufft_opts &opts, cufinufft_plan* d_plan);
int cuspread3d_idriven(int nf1, int nf2, int nf3, int M, 
		const cufinufft_opts opts,cufinufft_plan *d_mem);
int cuspread3d_subprob_prop(int nf1, int nf2, int nf3, int M, 
		const cufinufft_opts opts, cufinufft_plan *d_plan);
int cuspread3d_gather_prop(int nf1, int nf2, int nf3, int M, 
		const cufinufft_opts opts, cufinufft_plan *d_plan);
int cuspread3d_subprob(int nf1, int nf2, int nf3, int M, 
		const cufinufft_opts opts, cufinufft_plan *d_mem);

#endif
