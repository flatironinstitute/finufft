#ifndef __SPREAD1D_H__
#define __SPREAD1D_H__

#include "utils.h"

struct spread_opts {      // see cnufftspread:setup_spreader for defaults.
  int nspread;            // w, the kernel width in grid pts
  FLT upsampfac;          // sigma, upsampling factor, default 2.0
  // ES kernel specific...
  FLT ES_beta;
  FLT ES_halfwidth;
  FLT ES_c;
  
  // for output driven
  int bin_size_x;
  int bin_size_y;
  int use_thrust;
};

//Kernels for 1D codes
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

//Kernels for 2D codes
__global__
void CalcBinSize_2d(int M, int nf1, int nf2, int  bin_size_x, int bin_size_y, int nbinx,
                    int nbiny, int* bin_size, FLT *x, FLT *y, int* sortidx);
__global__
void CalcBinSize_noghost_2d(int M, int nf1, int nf2, int  bin_size_x, int bin_size_y, int nbinx,
                            int nbiny, int* bin_size, FLT *x, FLT *y, int* sortidx);
__global__
void FillGhostBin_2d(int nbinx, int nbiny, int*bin_size);
__global__
void BinsStartPts_2d(int M, int totalnumbins, int* bin_size, int* bin_startpts);
__global__
void prescan(int n, int* bin_size, int* bin_startpts, int* scanblock_sum);
__global__
void uniformUpdate(int n, int* data, int* buffer);

__global__
void PtsRearrage_2d(int M, int nf1, int nf2, int bin_size_x, int bin_size_y, int nbinx, int nbiny,
                    int* bin_startpts, int* sortidx, FLT *x, FLT *x_sorted,
                    FLT *y, FLT *y_sorted, gpuComplex *c, gpuComplex *c_sorted);
__global__
void PtsRearrage_noghost_2d(int M, int nf1, int nf2, int bin_size_x, int bin_size_y, int nbinx, int nbiny,
                            int* bin_startpts, int* sortidx, FLT *x, FLT *x_sorted,
                            FLT *y, FLT *y_sorted, gpuComplex *c, gpuComplex *c_sorted);
__global__
void Spread_2d_Odriven(int nbin_block_x, int nbin_block_y, int nbinx, int nbiny, int *bin_startpts,
                       FLT *x_sorted, FLT *y_sorted, gpuComplex *c_sorted, gpuComplex *fw, int ns,
                       int nf1, int nf2, FLT es_c, FLT es_beta, int fw_width);
__global__
void Spread_2d_Idriven(FLT *x, FLT *y, gpuComplex *c, gpuComplex *fw, int M, const int ns,
                       int nf1, int nf2, FLT es_c, FLT es_beta, int fw_width);
__global__
void Spread_2d_Hybrid(FLT *x, FLT *y, gpuComplex *c, gpuComplex *fw, int M, const int ns,
                      int nf1, int nf2, FLT es_c, FLT es_beta, int fw_width, int* binstartpts,
                      int* bin_size, int bin_size_x, int bin_size_y);
__global__
void CreateSortIdx (int M, int nf1, int nf2, FLT *x, FLT *y, int* sortidx);
__global__ 
void CreateIndex(int* index, int nelem);
__global__
void Gather(int nelem, int* index, FLT* x, FLT* y, gpuComplex* c,
           FLT* xsorted, FLT* ysorted, gpuComplex* csorted);


int cnufftspread2d_gpu_odriven(int nf1, int nf2, CPX* h_fw, int M, FLT *h_kx,
                               FLT *h_ky, CPX* h_c, spread_opts opts);
int cnufftspread2d_gpu_idriven(int nf1, int nf2, CPX* h_fw, int M, FLT *h_kx,
                               FLT *h_ky, CPX *h_c, spread_opts opts);
int cnufftspread2d_gpu_idriven_sorted(int nf1, int nf2, CPX* h_fw, int M, FLT *h_kx,
                                      FLT *h_ky, CPX* h_c, spread_opts opts);
int cnufftspread2d_gpu_hybrid(int nf1, int nf2, CPX* h_fw, int M, FLT *h_kx,
                              FLT *h_ky, CPX *h_c, spread_opts opts);

#endif
