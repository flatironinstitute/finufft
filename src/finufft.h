#ifndef FINUFFT_H
#define FINUFFT_H

#include "utils.h"
#include <fftw3.h>

struct nufft_opts {  // static init: this sets default opts for NUFFT alg:
  FLT R = 2.0;            // kernel-dep upsampling ratio (for experts)
  int debug = 0;          // 0: silent, 1: text timing output, 2: spread info
  int spread_debug = 0;   // passed to spread_opts debug: 0,1 or 2
  int spread_sort = 1;    // passed to spread_opts sort: 0 or 1
  int fftw = FFTW_ESTIMATE; // use FFTW_MEASURE for slow first call, fast rerun
  int modeord = 0;        // 0: CMCL mode ordering; 1: FFT-style ordering (not yet implemented)
};

// library provides...
int finufft1d1(INT nj,FLT* xj,CPX* cj,int iflag,FLT eps,INT ms,
	       CPX* fk, nufft_opts opts);
int finufft1d2(INT nj,FLT* xj,CPX* cj,int iflag,FLT eps,INT ms,
	       CPX* fk, nufft_opts opts);
int finufft1d3(INT nj,FLT* x,CPX* c,int iflag,FLT eps,INT nk, FLT* s, CPX* f, nufft_opts opts);

int finufft2d1(INT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	       INT ms, INT mt, CPX* fk, nufft_opts opts);
int finufft2d2(INT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	       INT ms, INT mt, CPX* fk, nufft_opts opts);
int finufft2d3(INT nj,FLT* x,FLT *y,CPX* cj,int iflag,FLT eps,INT nk, FLT* s, FLT* t, CPX* fk, nufft_opts opts);

int finufft3d1(INT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,FLT eps,
	       INT ms, INT mt, INT mu, CPX* fk, nufft_opts opts);
int finufft3d2(INT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,FLT eps,
	       INT ms, INT mt, INT mu, CPX* fk, nufft_opts opts);
int finufft3d3(INT nj,FLT* x,FLT *y,FLT *z, CPX* cj,int iflag,
	       FLT eps,INT nk,FLT* s, FLT* t, FLT *u,
	       CPX* fk, nufft_opts opts);

#endif
