#ifndef CNUFFTSPREAD_H
#define CNUFFTSPREAD_H

#include <stdlib.h>
#include <stdio.h>
#include "utils.h"

// choose if handle huge I/O array sizes (>2^31)
#define BIGINT int64_t
// #define BIGINT int32_t

// Note -std=c++11 is needed to avoid warning for static initialization here:
struct spread_opts {
  int kernel_type=1;       // 1: Kaiser--Bessel.
  int nspread=6;           // defaults are not too meaningful here, err 1e-5?
// opts controlling spreading method (indep of kernel type)...
  int spread_direction=1;  // 1 means spread NU->U, 2 means interpolate U->NU
  bool sort_data=true;     // controls method
  int checkerboard=0;      // controls method
  int debug=0;             // text output
// KB kernel specific...
  double KB_fac1=1;
  double KB_fac2=1;
  void set_W_and_beta();   // must be called before using KB kernel
  double KB_W;             // derived params, only experts should change...
  double KB_beta;
};

int cnufftspread(long N1, long N2, long N3, double *data_uniform,
		 long M, double *kx, double *ky, double *kz,
		 double *data_nonuniform, spread_opts opts);

void evaluate_kernel(int len, double *x, double *values, spread_opts opts);
int set_KB_opts_from_kernel_params(spread_opts &opts,double *kernel_params);
int set_KB_opts_from_eps(spread_opts &opts,double eps);
int get_kernel_params_for_eps(double *kernel_params,double eps);

#endif // CNUFFTSPREAD_H
