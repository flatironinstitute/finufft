#ifndef FINUFFT_H
#define FINUFFT_H

#include "utils.h"
#include "../spread.h"

struct nufft_opts {   // see common/finufft_default_opts() for defaults
  int debug;          // 0: silent, 1: text basic timing output
  int spread_debug;   // passed to spread_opts, 0 (no text) 1 (some) or 2 (lots)
  int spread_sort;    // passed to spread_opts, 0 (don't sort) 1 (do) or 2 (heuristic)
  int spread_kerevalmeth; // "     spread_opts, 0: exp(sqrt()), 1: Horner ppval (faster)
  int spread_kerpad;  // passed to spread_opts, 0: don't pad to mult of 4, 1: do
  int chkbnds;        // 0: don't check if input NU pts in [-3pi,3pi], 1: do
  int fftw;           // 0:FFTW_ESTIMATE, or 1:FFTW_MEASURE (slow plan but faster)
  int modeord;        // 0: CMCL-style increasing mode ordering (neg to pos), or
                      // 1: FFT-style mode ordering (affects type-1,2 only)
  int many_seq;       // 0: simultaneously do nufft on all data
                      // 1: sequentially run through the data
  int nsimul;
  FLT upsampfac;      // upsampling ratio sigma, either 2.0 (standard) or 1.25 (small FFT)
};

// library provides...
void finufft_default_opts(nufft_opts &o);

int finufft2d1_cpu(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	           BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);
int finufft2d1_gpu(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	           BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);

#endif
