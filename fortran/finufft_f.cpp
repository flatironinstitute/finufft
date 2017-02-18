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

void finufft2d1_f_(int *nj,double* xj,double *yj,dcomplex* cj,int *iflag,
		   double *eps, int *ms, int *mt, dcomplex* fk, int *ier)
{
  nufft_opts opts;
  *ier = finufft2d1(*nj,xj,yj,cj,*iflag,*eps,*ms,*mt,fk,opts);
}

void finufft2d2_f_(int *nj,double* xj,double *yj,dcomplex* cj,int *iflag,
		   double *eps, int *ms, int *mt, dcomplex* fk, int *ier)
{
  nufft_opts opts;
  *ier = finufft2d2(*nj,xj,yj,cj,*iflag,*eps,*ms,*mt,fk,opts);
}

void finufft2d3_f_(int *nj,double* xj,double* yj, dcomplex* cj,int *iflag,
		   double *eps, int *nk, double* s, double* t, dcomplex* fk,
		   int *ier)
{
  nufft_opts opts;
  *ier = finufft2d3(*nj,xj,yj,cj,*iflag,*eps,*nk,s,t,fk,opts);
}

void finufft3d1_f_(int *nj,double* xj,double *yj,double* zj,dcomplex* cj,
		   int *iflag, double *eps, int *ms, int* mt, int* mu,
		   dcomplex* fk, int *ier)
{
  nufft_opts opts;
  *ier = finufft3d1(*nj,xj,yj,zj,cj,*iflag,*eps,*ms,*mt,*mu,fk,opts);
}

void finufft3d2_f_(int *nj,double* xj,double *yj,double* zj,dcomplex* cj,
		   int *iflag, double *eps, int *ms, int* mt, int* mu,
		   dcomplex* fk, int *ier)
{
  nufft_opts opts;
  *ier = finufft3d2(*nj,xj,yj,zj,cj,*iflag,*eps,*ms,*mt,*mu,fk,opts);
}

void finufft3d3_f_(int *nj,double* xj,double* yj, double*zj, dcomplex* cj,
		   int *iflag, double *eps, int *nk, double* s, double* t,
		   double* u, dcomplex* fk, int *ier)
{
  nufft_opts opts;
  *ier = finufft3d3(*nj,xj,yj,zj,cj,*iflag,*eps,*nk,s,t,u,fk,opts);
}
