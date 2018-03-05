#ifndef FINUFFT_H
#define FINUFFT_H

#include "utils.h"

struct nufft_opts {  // see common/finufft_default_opts() for defaults
  int debug;          // 0: silent, 1: text basic timing output
  int spread_debug;   // passed to spread_opts debug: 0,1 or 2
  int spread_sort;    // passed to spread_opts sort: 0,1 or 2
  int chkbnds;        // 0: don't check if input NU pts in [-3pi,3pi], 1: do
  int fftw;    // FFTW_ESTIMATE, or FFTW_MEASURE for slow first call, fast rerun
  int modeord;        // 0: CMCL-style increasing mode ordering, or
                      // 1: FFT-style mode ordering (affects type-1,2 only
  FLT R;              // kernel-dep upsampling ratio (don't change it!)
};

// library provides...
void finufft_default_opts(nufft_opts &o);
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
