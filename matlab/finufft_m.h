// declare interfaces that matlab/octave can call via MEX

#ifndef FINUFFT_M_H
#define FINUFFT_M_H

// for the C++ complex type:
#include <complex>

// for matlab interface (always double prec for now) we use this left-over name,
// which happens to be the same as what Mwrap calls double complex (but
// this is not required):
typedef std::complex<double> dcomplex;

// library provides (facing MEX)....
int finufft1d1m(double nj,double* xj,dcomplex* cj,int iflag,double eps,double ms, dcomplex* fk, double* opts);
int finufft1d2m(double nj,double* xj,dcomplex* cj,int iflag,double eps,double ms, dcomplex* fk, double* opts);
int finufft1d3m(double nj,double* xj,dcomplex* cj,int iflag,double eps,double nk, double* s, dcomplex* f, double * opts);

int finufft2d1m(double nj,double* xj,double* yj, dcomplex* cj,int iflag,double eps,double ms, double mt, dcomplex* fk, double* opts);
int finufft2d1manym(double ndata,double nj,double* xj,double* yj, dcomplex* cj,int iflag,double eps,double ms, double mt, dcomplex* fk, double* opts);
int finufft2d2m(double nj,double* xj,double* yj, dcomplex* cj,int iflag,double eps,double ms, double mt, dcomplex* fk, double* opts);
int finufft2d2manym(double ndata,double nj,double* xj,double* yj, dcomplex* cj,int iflag,double eps,double ms, double mt, dcomplex* fk, double* opts);
int finufft2d3m(double nj,double* xj,double* yj, dcomplex* cj,int iflag,double eps,double nk, double* s, double* t, dcomplex* f, double *opts);

int finufft3d1m(double nj,double* xj,double* yj, double*zj, dcomplex* cj,int iflag,double eps,double ms, double mt, double mu, dcomplex* fk, double* opts);
int finufft3d2m(double nj,double* xj,double* yj,double*zj, dcomplex* cj,int iflag,double eps,double ms, double mt, double mu, dcomplex* fk, double* opts);
int finufft3d3m(double nj,double* xj,double* yj,double*zj, dcomplex* cj,int iflag,double eps,double nk, double* s, double* t, double* u, dcomplex* f, double* opts);

#endif
