#ifndef TWOPISPREAD_H
#define TWOPISPREAD_H

#include "utils.h"
#include "cnufftspread.h"

int twopispread1d(long nf1,double *fw,BIGINT nj,double* xj,double* cj,
		  spread_opts opts);
int twopispread2d(long nf1,long nf2, double *fw,BIGINT nj,double* xj,
		  double *yj,double* cj,spread_opts opts);
int twopispread3d(long nf1,long nf2,long nf3,double *fw,BIGINT nj,double* xj,
		  double *yj,double* zj,double* cj,spread_opts opts);

#endif
