#include "finufft_f.h"
#include "../src/finufft.h"

// wrappers for calling FINUFFT from fortran.
// Note the trailing underscore which is not present in the fortran name.
// Barnett 2/17/17

// integer*4 for the sizes for now

void finufft1d1_f_(int *nj,double* xj,dcomplex* cj,int *iflag, double *eps,
		  int *ms, dcomplex* fk, int *ier)
{
  nufft_opts opts;  
  *ier = finufft1d1(*nj,xj,cj,*iflag,*eps,*ms,fk,opts);
}

void finufft1d2_f_(int *nj,double* xj,dcomplex* cj,int *iflag, double *eps,
		  int *ms, dcomplex* fk, int *ier)
{
  nufft_opts opts;  
  *ier = finufft1d2(*nj,xj,cj,*iflag,*eps,*ms,fk,opts);
}

void finufft1d3_f_(int *nj,double* xj,dcomplex* cj,int *iflag, double *eps,
		  int *nk, double* s, dcomplex* fk, int *ier)
{
  nufft_opts opts;  
  *ier = finufft1d3(*nj,xj,cj,*iflag,*eps,*nk,s,fk,opts);
}
