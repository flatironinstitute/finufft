#include "../src/finufft.h"

// wrappers for calling FINUFFT from fortran
// Barnett 2/17/17

// integer*4 for the sizes for now

// ** EXTERN C CRAP


void finufft1d1_f(int *nj,double* xj,dcomplex* cj,int *iflag, double *eps,
		  int *ms, dcomplex* fk, int *ier)
{
  nufft_opts opts;  
  *ier = finufft1d1(*nj,xj,(double*)cj,*iflag,*eps,*ms,(double*)fk,opts);
}

void finufft1d2_f(int *nj,double* xj,dcomplex* cj,int *iflag, double *eps,
		  int *ms, dcomplex* fk, int *ier)
{
  nufft_opts opts;  
  *ier = finufft1d2(*nj,xj,(double*)cj,*iflag,*eps,*ms,(double*)fk,opts);
}

void finufft1d3_f(int *nj,double* xj,dcomplex* cj,int *iflag, double *eps,
		  int *nk, double* s, dcomplex* fk, int *ier)
{
  nufft_opts opts;  
  *ier = finufft1d3(*nj,xj,(double*)cj,*iflag,*eps,*nk,s,(double*)fk,opts);
}
