#ifndef CNUFFTSPREAD_H
#define CNUFFTSPREAD_H

#include <stdlib.h>
#include <stdio.h>

// choose if handle huge I/O array sizes (>2^31)  todo: use this
#define BIGINT long long
// #define BIGINT long


struct cnufftspread_opts {  // -std=c++11 is needed to avoid these giving warnings:
  int nspread=6;
  double KB_fac1=1;
  double KB_fac2=1;
  int spread_direction=1;     // 1 means spread NU->U, 2 means interpolate U->NU
  bool sort_data=true;
  int checkerboard=0;
  int debug=0;
  void set_W_and_beta();   // must be called before spreading.
  double KB_W;             // derived parameters, for experts only
  double KB_beta;

};

int cnufftspread(long N1, long N2, long N3, double *data_uniform,
		 long M, double *kx, double *ky, double *kz, double *data_nonuniform,
		 cnufftspread_opts opts
		 );

void evaluate_kernel(int len, double *x, double *values, cnufftspread_opts opts);
void set_kb_opts_from_kernel_params(cnufftspread_opts &opts,double *kernel_params);
void set_kb_opts_from_eps(cnufftspread_opts &opts,double eps);


// MATLAB interface

/*
 * MCWRAP [ COMPLEX Y[N,N,N] ] = cnufftspread_type1(N,kx[M,1],ky[M,1],kz[M,1],COMPLEX X[M,1],kernel_params[4,1])
 * SET_INPUT M = size(kx,1)
 * SOURCES cnufftspread.cpp ../contrib/besseli.cpp
 * HEADERS cnufftspread.h ../contrib/besseli.h
 */

void cnufftspread_type1(int N,double *Y,int M,double *kx,double *ky,double *kz,double *X,double *kernel_params);
// todo: add ier as type or arg ptr

class CNTime {
 public:
  void start();
  int restart();
  int elapsed();
  double elapsedsec();
 private:
  struct timeval initial;
};

#endif // CNUFFTSPREAD_H
