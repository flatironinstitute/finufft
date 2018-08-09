#ifndef __SPREAD_H__
#define __SPREAD_H__

#include "finufft/utils.h"

#define MAX_NSPREAD 16
#define RESCALE(x,N,p) (p ? \
                       ((x*M_1_2PI + (x<-PI ? 1.5 : (x>PI ? -0.5 : 0.5)))*N) : \
                       (x<0 ? x+N : (x>N ? x-N : x)))

struct spread_opts {      // see cnufftspread:setup_spreader for defaults.
  int nspread;            // w, the kernel width in grid pts
  int spread_direction;   // 1 means spread NU->U, 2 means interpolate U->NU
  int pirange;            // 0: coords in [0,N), 1 coords in [-pi,pi)
  int chkbnds;            // 0: don't check NU pts are in range; 1: do
  int sort;               // 0: don't sort NU pts, 1: do, 2: heuristic choice
  int kerevalmeth;        // 0: exp(sqrt()), old, or 1: Horner ppval, fastest
  int kerpad;             // 0: no pad to mult of 4, 1: do (helps i7 kereval=0)
  int sort_threads;       // 0: auto-choice, >0: fix number of sort threads
  int max_subproblem_size; // sets extra RAM per thread
  int flags;              // binary flags for timing only (may give wrong ans!)
  int debug;              // 0: silent, 1: small text output, 2: verbose
  FLT upsampfac;          // sigma, upsampling factor, default 2.0

  // ES kernel specific...
  FLT ES_beta;
  FLT ES_halfwidth;
  FLT ES_c;
  
  // CUDA: for output driven
  int method;
  int bin_size_x;
  int bin_size_y;
  int Horner;
  int maxsubprobsize;
  int nthread_x;
  int nthread_y;
};

struct spread_devicemem {
  int byte_now;
  FLT *fwkerhalf1;
  FLT *fwkerhalf2;

  FLT *kx;
  FLT *ky;
  CUCPX *c;
  CUCPX *fw;
  CUCPX *fk;
  
  FLT *kxsorted;
  FLT *kysorted;
  CUCPX *csorted;

  int *sortidx;
  int *binsize;
  int *binstartpts;
  int *numsubprob;
  int *subprob_to_bin;
  int *idxnupts;
  int *subprobstartpts;

  void *temp_storage;
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
void CalcBinSize_noghost_2d(int M, int nf1, int nf2, int  bin_size_x, int bin_size_y, int nbinx,
                    int nbiny, int* bin_size, FLT *x, FLT *y, int* sortidx);
__global__
void uniformUpdate(int n, int* data, int* buffer);
__global__
void CalcInvertofGlobalSortIdx_2d(int M, int bin_size_x, int bin_size_y, int nbinx,
                                  int nbiny, int* bin_startpts, int* sortidx,
                                  FLT *x, FLT *y, int* index);
__global__
void PtsRearrage_noghost_2d(int M, int nf1, int nf2, int bin_size_x, int bin_size_y, int nbinx,
                int nbiny, int* bin_startpts, int* sortidx, FLT *x, FLT *x_sorted,
                FLT *y, FLT *y_sorted, CUCPX *c, CUCPX *c_sorted);
__global__
void Spread_2d_Odriven(int nbin_block_x, int nbin_block_y, int nbinx, int nbiny, int *bin_startpts,
                       FLT *x_sorted, FLT *y_sorted, CUCPX *c_sorted, CUCPX *fw, int ns,
                       int nf1, int nf2, FLT es_c, FLT es_beta, int fw_width);
__global__
void Spread_2d_Idriven(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
                       int nf1, int nf2, FLT es_c, FLT es_beta, int fw_width);
__global__
void Spread_2d_Idriven_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
                              int nf1, int nf2, FLT es_c, FLT es_beta, int fw_width);
__global__
void Spread_2d_Hybrid(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
                      int nf1, int nf2, FLT es_c, FLT es_beta, int fw_width, int* binstartpts,
                      int* bin_size, int bin_size_x, int bin_size_y);
__global__
void Spread_2d_Simple(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
                      int nf1, int nf2, FLT es_c, FLT es_beta, int fw_width, int bin_size,
                      int bin_size_x, int bin_size_y, int binx, int biny);
__global__
void CalcSubProb_2d(int* bin_size, int* num_subprob, int maxsubprobsize, int numbins);
__global__
void MapBintoSubProb_2d(int* d_subprob_to_bin, int* d_subprobstartpts, int* d_numsubprob,
                        int numbins);
__global__
void Spread_2d_Subprob(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
                          int nf1, int nf2, FLT es_c, FLT es_beta, FLT sigma, int fw_width, int* binstartpts,
                          int* bin_size, int bin_size_x, int bin_size_y, int* subprob_to_bin,
                          int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx, int nbiny,
                          int* idxnupts);


int cnufftspread2d_gpu(int ms, int mt, int nf1, int nf2, CPX* h_fw, int M, FLT *h_kx,
                       FLT *h_ky, CPX* h_c, spread_opts opts, spread_devicemem *dmem);
int cnufftspread2d_gpu_idriven(int nf1, int nf2, int fw_width, int M, spread_opts opts, 
                               spread_devicemem *d_mem);
int cnufftspread2d_gpu_idriven_sorted(int nf1, int nf2, int fw_width, int M, spread_opts opts,
                               spread_devicemem *d_mem);
int cnufftspread2d_gpu_hybrid(int nf1, int nf2, int fw_width, int M, spread_opts opts,
                               spread_devicemem *d_mem);
int cnufftspread2d_gpu_subprob(int nf1, int nf2, int fw_width, int M, spread_opts opts,
                               spread_devicemem *d_mem);
int cnufftspread2d_gpu_simple(int nf1, int nf2, int fw_width, CUCPX* d_fw, int M, FLT *d_kx,
                              FLT *d_ky, CUCPX *d_c, spread_opts opts, int binx, int biny);

int cnufft_allocgpumemory(int ms, int mt, int nf1, int nf2, int M, int* fw_width, spread_opts opts, spread_devicemem *d_mem);
int cnufft_copycpumem_to_gpumem(int M, FLT *h_kx, FLT* h_ky, CPX *h_c, int nf1, int nf2, FLT* h_fwkerhalf1, 
                                FLT* h_fwkerhalf2, spread_devicemem *d_mem);
int cnufft_copygpumem_to_cpumem_fw(int nf1, int nf2, int fw_width, CPX* h_fw, spread_devicemem *d_mem);
void cnufft_free_gpumemory(spread_opts opts, spread_devicemem *d_mem);
#endif
