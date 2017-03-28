#ifndef FINUFFT_H
#define FINUFFT_H

#include "utils.h"

struct nufft_opts {  // this sets default opts for NUFFT alg:
  double R = 2.0;         // kernel-dep upsampling ratio (only experts change)
  int debug = 0;          // 0: silent, 1: text timing output, 2: spread info
  int spread_debug = 0;   // passed to spread_opts debug: 0,1 or 2
  int spread_sort = 1 ;   // passed to spread_opts sort: 0 or 1
  BIGINT maxnalloc = (BIGINT)1e9;   // largest size of internal array to malloc
};

// library provides...
int finufft1d1(BIGINT nj,double* xj,dcomplex* cj,int iflag,double eps,BIGINT ms,
	       dcomplex* fk, nufft_opts opts);
int finufft1d2(BIGINT nj,double* xj,dcomplex* cj,int iflag,double eps,BIGINT ms,
	       dcomplex* fk, nufft_opts opts);
int finufft1d3(BIGINT nj,double* x,dcomplex* c,int iflag,double eps,BIGINT nk, double* s, dcomplex* f, nufft_opts opts);

int finufft2d1(BIGINT nj,double* xj,double *yj,dcomplex* cj,int iflag,double eps,
	       BIGINT ms, BIGINT mt, dcomplex* fk, nufft_opts opts);
int finufft2d2(BIGINT nj,double* xj,double *yj,dcomplex* cj,int iflag,double eps,
	       BIGINT ms, BIGINT mt, dcomplex* fk, nufft_opts opts);
int finufft2d3(BIGINT nj,double* x,double *y,dcomplex* cj,int iflag,double eps,BIGINT nk, double* s, double* t, dcomplex* fk, nufft_opts opts);

int finufft3d1(BIGINT nj,double* xj,double *yj,double *zj,dcomplex* cj,int iflag,double eps,
	       BIGINT ms, BIGINT mt, BIGINT mu, dcomplex* fk, nufft_opts opts);
int finufft3d2(BIGINT nj,double* xj,double *yj,double *zj,dcomplex* cj,int iflag,double eps,
	       BIGINT ms, BIGINT mt, BIGINT mu, dcomplex* fk, nufft_opts opts);
int finufft3d3(BIGINT nj,double* x,double *y,double *z, dcomplex* cj,int iflag,
	       double eps,BIGINT nk,double* s, double* t, double *u,
	       dcomplex* fk, nufft_opts opts);

#endif
