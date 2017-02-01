#ifndef TWOPISPREAD_H
#define TWOPISPREAD_H

#include "cnufftspread.h"
#include <math.h>

int twopispread1d(long nf1,double *fw,BIGINT nj,double* xj,double* cj,
		  int dir,double* params, int debug);
int twopispread2d(long nf1,long nf2, double *fw,BIGINT nj,double* xj,
		  double *yj,double* cj,int dir,double* params, int debug);
int twopispread3d(long nf1,long nf2,long nf3,double *fw,BIGINT nj,double* xj,
		  double *yj,double* zj,double* cj,int dir,double* params,
		  int debug);

#endif
