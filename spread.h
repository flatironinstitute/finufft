#ifndef __SPREAD1D_H__
#define __SPREAD1D_H__

#include "utils.h"

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
void FillGhostBin_2d(int bin_size_x, int bin_size_y, int nbinx, int nbiny, int*bin_size);

__global__
void BinsStartPts_2d(int M, int totalnumbins, int* bin_size, int* bin_startpts);
__global__
void prescan(int n, int* bin_size, int* bin_startpts);

__global__
void PtsRearrage_2d(int M, int nf1, int nf2, int bin_size_x, int bin_size_y, int nbinx, int nbiny,
                    int* bin_startpts, int* sortidx, FLT *x, FLT *x_sorted,
                    FLT *y, FLT *y_sorted, FLT *c, FLT *c_sorted);
__global__
void Spread_2d_Odriven(int nbin_block_x, int nbin_block_y, int nbinx, int nbiny, int *bin_startpts,
                       FLT *x_sorted, FLT *y_sorted, FLT *c_sorted, FLT *fw, int ns,
                       int nf1, int nf2, FLT es_c, FLT es_beta);
__global__
void Spread_2d_Idriven(FLT *x_sorted, FLT *y_sorted, FLT *c_sorted, FLT *fw, int M, int ns,
                       int nf1, int nf2, FLT es_c, FLT es_beta);

int cnufftspread2d_gpu_odriven(int nf1, int nf2, FLT* h_fw, int M, FLT *h_kx,
                               FLT *h_ky, FLT *h_c, int bin_size_x, int bin_size_y);

int cnufftspread2d_gpu_idriven(int nf1, int nf2, FLT* h_fw, int M, FLT *h_kx,
                               FLT *h_ky, FLT *h_c);
#endif
