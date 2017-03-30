// declare interfaces that matlab/octave can call via MEX

#ifndef FINUFFT_M_H
#define FINUFFT_M_H

#include "../src/finufft.h"

int finufft1d1m(double nj,double* xj,dcomplex* cj,int iflag,double eps,double ms, dcomplex* fk,int debug,double maxnalloc, int nthreads, int spread_sort);
int finufft1d2m(double nj,double* xj,dcomplex* cj,int iflag,double eps,double ms, dcomplex* fk,int debug,double maxnalloc, int nthreads, int spread_sort);
int finufft1d3m(double nj,double* xj,dcomplex* cj,int iflag,double eps,double nk, double* s, dcomplex* f, int debug,double maxnalloc, int nthreads, int spread_sort);

int finufft2d1m(double nj,double* xj,double* yj, dcomplex* cj,int iflag,double eps,double ms, double mt, dcomplex* fk,int debug,double maxnalloc, int nthreads, int spread_sort);
int finufft2d2m(double nj,double* xj,double* yj,dcomplex* cj,int iflag,double eps,double ms, double mt, dcomplex* fk,int debug,double maxnalloc, int nthreads, int spread_sort);
int finufft2d3m(double nj,double* xj,double* yj,dcomplex* cj,int iflag,double eps,double nk, double* s, double* t, dcomplex* f, int debug,double maxnalloc, int nthreads, int spread_sort);

int finufft3d1m(double nj,double* xj,double* yj, double*zj, dcomplex* cj,int iflag,double eps,double ms, double mt, double mu, dcomplex* fk,int debug,double maxnalloc, int nthreads, int spread_sort);
int finufft3d2m(double nj,double* xj,double* yj,double*zj, dcomplex* cj,int iflag,double eps,double ms, double mt, double mu, dcomplex* fk,int debug,double maxnalloc, int nthreads, int spread_sort);
int finufft3d3m(double nj,double* xj,double* yj,double*zj, dcomplex* cj,int iflag,double eps,double nk, double* s, double* t, double* u, dcomplex* f, int debug,double maxnalloc, int nthreads, int spread_sort);

#endif
