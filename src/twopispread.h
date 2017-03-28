#ifndef TWOPISPREAD_H
#define TWOPISPREAD_H

#include "utils.h"
#include "cnufftspread.h"

int twopispread1d(BIGINT nf1,dcomplex *fw,BIGINT nj,double* xj,dcomplex* cj,
		  spread_opts opts);
int twopispread2d(BIGINT nf1,BIGINT nf2, dcomplex *fw,BIGINT nj,double* xj,
		  double *yj,dcomplex* cj,spread_opts opts);
int twopispread3d(BIGINT nf1,BIGINT nf2,BIGINT nf3,dcomplex *fw,BIGINT nj,double* xj,
		  double *yj,double* zj,dcomplex* cj,spread_opts opts);

#endif
