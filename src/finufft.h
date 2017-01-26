#ifndef FINUFFT_H
#define FINUFFT_H

#include "utils.h"
#include "cnufftspread.h"
#include "twopispread.h"
#include <fftw3.h>

struct nufft_opts {
  double R = 2.0;        // upsampling ratio
  int debug = 0;             // 0: silent, 1: text output
  int spread_debug = 0;      // passed to spread_opts debug
};

#include "common.h"   // here since needs nufft_opts already defined

int finufft1d1(BIGINT nj,double* xj,double* cj,int iflag,double eps,BIGINT ms,
	       double* fk, nufft_opts opts);
int finufft1d2(BIGINT nj,double* xj,double* cj,int iflag,double eps,BIGINT ms,
	       double* fk, nufft_opts opts);
int finufft1d3(BIGINT nj,double* x,dcomplex* c,int iflag,BIGINT nk, double* s, dcomplex* f, nufft_opts opts);

#endif
