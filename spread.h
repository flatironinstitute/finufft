#ifndef __SPREAD1D_H__
#define __SPREAD1D_H__

//Kernels for 1D codes
__global__
void CalcBinSize_1d(int M, int nf1, int  bin_size_x, int nbinx,
                    int* bin_size, double *x, int* sortidx);
__global__
void FillGhostBin_1d(int bin_size_x, int nbinx, int*bin_size);

__global__
void BinsStartPts_1d(int M, int totalnumbins, int* bin_size, int* bin_startpts);

__global__
void PtsRearrage_1d(int M, int nf1, int bin_size_x, int nbinx,
                    int* bin_startpts, int* sortidx, double *x, double *x_sorted,
                    double *c, double *c_sorted);
__global__
void Spread_1d(int nbin_block_x, int nbinx, int *bin_startpts,
               double *x_sorted, double *c_sorted, double *fw, int ns,
               int nf1, double es_c, double es_beta);

//Kernels for 2D codes
__global__
void CalcBinSize_2d(int M, int nf1, int nf2, int  bin_size_x, int bin_size_y, int nbinx,
                    int nbiny, int* bin_size, double *x, double *y, int* sortidx);

__global__
void FillGhostBin_2d(int bin_size_x, int bin_size_y, int nbinx, int nbiny, int*bin_size);

__global__
void BinsStartPts_2d(int M, int totalnumbins, int* bin_size, int* bin_startpts);

__global__
void PtsRearrage_2d(int M, int nf1, int nf2, int bin_size_x, int bin_size_y, int nbinx, int nbiny,
                    int* bin_startpts, int* sortidx, double *x, double *x_sorted,
                    double *y, double *y_sorted, double *c, double *c_sorted);
__global__
void Spread_2d(int nbin_block_x, int nbin_block_y, int nbinx, int nbiny, int *bin_startpts,
               double *x_sorted, double *y_sorted, double *c_sorted, double *fw, int ns,
               int nf1, int nf2, double es_c, double es_beta);
#endif
