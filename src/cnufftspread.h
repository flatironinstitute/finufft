#ifndef CNUFFTSPREAD_H
#define CNUFFTSPREAD_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"

#define MAX_NSPREAD 16       // upper bound on nspread, also needed by common

// Note -std=c++11 is needed to avoid warning for static initialization here:
struct spread_opts {
  int nspread=6;           // defaults are not too meaningful here
// opts controlling spreading method (indep of kernel)...
  int spread_direction=1;  // 1 means spread NU->U, 2 means interpolate U->NU
  bool sort_data=true;     // controls method
  int checkerboard=0;      // controls method
  int debug=0;             // text output
// ES kernel specific...
  double ES_beta;
  double ES_halfwidth;
  double ES_c;
};

int cnufftspread(BIGINT N1, BIGINT N2, BIGINT N3, double *data_uniform,
		 BIGINT M, double *kx, double *ky, double *kz,
		 double *data_nonuniform, spread_opts opts);
double evaluate_kernel(double x,const spread_opts &opts);
int setup_kernel(spread_opts &opts,double eps,double R);

#endif // CNUFFTSPREAD_H
