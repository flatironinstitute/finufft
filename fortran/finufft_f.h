#ifndef FINUFFT_F_H
#define FINUFFT_F_H

#include "../src/utils.h"

extern "C" {
void finufft1d1_f_(int *nj,double* xj,dcomplex* cj,int *iflag, double *eps,
		  int *ms, dcomplex* fk, int *ier);
void finufft1d2_f_(int *nj,double* xj,dcomplex* cj,int *iflag, double *eps,
		  int *ms, dcomplex* fk, int *ier);
void finufft1d3_f_(int *nj,double* xj,dcomplex* cj,int *iflag, double *eps,
		  int *nk, double* s, dcomplex* fk, int *ier);
void finufft2d1_f_(int *nj,double* xj,double *yj,dcomplex* cj,int *iflag,
		   double *eps, int *ms, int *mt, dcomplex* fk, int *ier);
void finufft2d2_f_(int *nj,double* xj,double *yj,dcomplex* cj,int *iflag,
		   double *eps, int *ms, int *mt, dcomplex* fk, int *ier);
void finufft2d3_f_(int *nj,double* xj,double* yj, dcomplex* cj,int *iflag,
		   double *eps, int *nk, double* s, double* t, dcomplex* fk,
		   int *ier);
void finufft3d1_f_(int *nj,double* xj,double *yj,double* zj,dcomplex* cj,
		   int *iflag, double *eps, int *ms, int *mt, int *mu,
		   dcomplex* fk, int *ier);
void finufft3d2_f_(int *nj,double* xj,double *yj,double* zj,dcomplex* cj,
		   int *iflag, double *eps, int *ms, int *mt, int *mu,
		   dcomplex* fk, int *ier);
void finufft3d3_f_(int *nj,double* xj,double* yj, double*zj, dcomplex* cj,
		   int *iflag, double *eps, int *nk, double* s, double* t,
		   double* u, dcomplex* fk, int *ier);
}

#endif
