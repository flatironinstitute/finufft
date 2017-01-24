#ifndef FINUFFT1D_H
#define FINUFFT1D_H

#include "cnufftspread.h"
#include "twopispread.h"
#include "common.h"
#include <fftw3.h>

int finufft1d1(BIGINT nj,double* xj,double* cj,int iflag,double eps,BIGINT ms,
	       double* fk);
int finufft1d2(BIGINT nj,double* xj,double* cj,int iflag,double eps,BIGINT ms,
	       double* fk);
int finufft1d3(BIGINT nj,double* xj,double* cj,int iflag,double eps,BIGINT mi,
	       double* fi);  // todo

#endif
