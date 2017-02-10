#ifndef FINUFFT_H
#define FINUFFT_H

#include "utils.h"
#include "cnufftspread.h"
#include "twopispread.h"
#include <fftw3.h>

struct nufft_opts {  // set default opts for NUFFT alg:
  double R = 2.0;            // upsampling ratio (can go down to around 1.8)
  int debug = 0;             // 0: silent, 1: text output
  int spread_debug = 0;      // passed to spread_opts debug
};

#include "common.h"   // here since needs nufft_opts already defined

int finufft1d1(BIGINT nj,double* xj,double* cj,int iflag,double eps,BIGINT ms,
	       double* fk, nufft_opts opts);
int finufft1d2(BIGINT nj,double* xj,double* cj,int iflag,double eps,BIGINT ms,
	       double* fk, nufft_opts opts);
int finufft1d3(BIGINT nj,double* x,double* c,int iflag,double eps,BIGINT nk, double* s, double* f, nufft_opts opts);

int finufft2d1(BIGINT nj,double* xj,double *yj,double* cj,int iflag,double eps,
	       BIGINT ms, BIGINT mt, double* fk, nufft_opts opts);
int finufft2d2(BIGINT nj,double* xj,double *yj,double* cj,int iflag,double eps,
	       BIGINT ms, BIGINT mt, double* fk, nufft_opts opts);
int finufft2d3(BIGINT nj,double* x,double *y,double* cj,int iflag,double eps,BIGINT nk, double* s, double* t, double* fk, nufft_opts opts);

int finufft3d1(BIGINT nj,double* xj,double *yj,double *zj,double* cj,int iflag,double eps,
	       BIGINT ms, BIGINT mt, BIGINT mu, double* fk, nufft_opts opts);
int finufft3d2(BIGINT nj,double* xj,double *yj,double *zj,double* cj,int iflag,double eps,
	       BIGINT ms, BIGINT mt, BIGINT mu, double* fk, nufft_opts opts);
int finufft2d3(BIGINT nj,double* x,double *y,double *z, double* cj,int iflag,
	       double eps,BIGINT nk,double* s, double* t, double *u,
	       double* fk, nufft_opts opts);

#endif
