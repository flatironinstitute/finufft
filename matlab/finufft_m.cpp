// C-style interface to FINUFFT library that is used for MWRAP interface.
// Note that no underscores can be used in the function names.
// We use doubles to rep big integers, since "long long" failed in mwrap.
// We tried reading MY_OMP_GET_MAX_THREADS() but since we're in MEX it always
// seems to be 1, so we don't use it.
// Barnett 3/24/17

#include "../src/finufft.h"
#include <stdio.h>

int finufft1d1m(double nj,double* xj,dcomplex* cj,int iflag,double eps,double ms, dcomplex* fk,int debug,double maxnalloc, int nthreads)
{
  nufft_opts opts;
  opts.debug = debug;
  if (maxnalloc>0) opts.maxnalloc = (BIGINT)maxnalloc;
  if (nthreads>0) MY_OMP_SET_NUM_THREADS(nthreads);
  return finufft1d1((BIGINT)nj,xj,cj,iflag,eps,(BIGINT)ms,fk,opts);
}

int finufft1d2m(double nj,double* xj,dcomplex* cj,int iflag,double eps,double ms, dcomplex* fk,int debug,double maxnalloc, int nthreads)
{
  nufft_opts opts;
  opts.debug = debug;
  if (maxnalloc>0) opts.maxnalloc = (BIGINT)maxnalloc;
  if (nthreads>0) MY_OMP_SET_NUM_THREADS(nthreads);
  return finufft1d2((BIGINT)nj,xj,cj,iflag,eps,(BIGINT)ms,fk,opts);
}

int finufft1d3m(double nj,double* xj,dcomplex* cj,int iflag,double eps,double nk, double* s, dcomplex* fk, int debug,double maxnalloc, int nthreads)
  {
  nufft_opts opts;
  opts.debug = debug;
  if (maxnalloc>0) opts.maxnalloc = (BIGINT)maxnalloc;
  if (nthreads>0) MY_OMP_SET_NUM_THREADS(nthreads);
  return finufft1d3((BIGINT)nj,xj,cj,iflag,eps,(BIGINT)nk,s,fk,opts);
}

int finufft2d1m(double nj,double* xj,double* yj, dcomplex* cj,int iflag,double eps,double ms, double mt, dcomplex* fk,int debug,double maxnalloc, int nthreads)
{
  nufft_opts opts;
  opts.debug = debug;
  if (maxnalloc>0) opts.maxnalloc = (BIGINT)maxnalloc;
  if (nthreads>0) MY_OMP_SET_NUM_THREADS(nthreads);
  return finufft2d1((BIGINT)nj,xj,yj,cj,iflag,eps,(BIGINT)ms,(BIGINT)mt,fk,opts);
}

int finufft2d2m(double nj,double* xj,double* yj,dcomplex* cj,int iflag,double eps,double ms, double mt, dcomplex* fk,int debug,double maxnalloc, int nthreads)
{
  nufft_opts opts;
  opts.debug = debug;
  if (maxnalloc>0) opts.maxnalloc = (BIGINT)maxnalloc;
  if (nthreads>0) MY_OMP_SET_NUM_THREADS(nthreads);
  return finufft2d2((BIGINT)nj,xj,yj,cj,iflag,eps,(BIGINT)ms,(BIGINT)mt,fk,opts);
}

int finufft2d3m(double nj,double* xj,double *yj,dcomplex* cj,int iflag,double eps,double nk, double* s, double* t, dcomplex* fk, int debug,double maxnalloc, int nthreads)
  {
  nufft_opts opts;
  opts.debug = debug;
  if (maxnalloc>0) opts.maxnalloc = (BIGINT)maxnalloc;
  if (nthreads>0) MY_OMP_SET_NUM_THREADS(nthreads);
  return finufft2d3((BIGINT)nj,xj,yj,cj,iflag,eps,(BIGINT)nk,s,t,fk,opts);
}

int finufft3d1m(double nj,double* xj,double* yj, double* zj, dcomplex* cj,int iflag,double eps,double ms, double mt, double mu, dcomplex* fk,int debug,double maxnalloc, int nthreads)
{
  nufft_opts opts;
  opts.debug = debug;
  if (maxnalloc>0) opts.maxnalloc = (BIGINT)maxnalloc;
  if (nthreads>0) MY_OMP_SET_NUM_THREADS(nthreads);
  return finufft3d1((BIGINT)nj,xj,yj,zj,cj,iflag,eps,(BIGINT)ms,(BIGINT)mt,(BIGINT)mu,fk,opts);
}

int finufft3d2m(double nj,double* xj,double* yj,double *zj,dcomplex* cj,int iflag,double eps,double ms, double mt, double mu, dcomplex* fk,int debug,double maxnalloc, int nthreads)
{
  nufft_opts opts;
  opts.debug = debug;
  if (maxnalloc>0) opts.maxnalloc = (BIGINT)maxnalloc;
  if (nthreads>0) MY_OMP_SET_NUM_THREADS(nthreads);
  return finufft3d2((BIGINT)nj,xj,yj,zj,cj,iflag,eps,(BIGINT)ms,(BIGINT)mt,(BIGINT)mu,fk,opts);
}

int finufft3d3m(double nj,double* xj,double *yj,double *zj,dcomplex* cj,int iflag,double eps,double nk, double* s, double* t, double *u, dcomplex* fk, int debug,double maxnalloc, int nthreads)
  {
  nufft_opts opts;
  opts.debug = debug;
  if (maxnalloc>0) opts.maxnalloc = (BIGINT)maxnalloc;
  if (nthreads>0) MY_OMP_SET_NUM_THREADS(nthreads);
  return finufft3d3((BIGINT)nj,xj,yj,zj,cj,iflag,eps,(BIGINT)nk,s,t,u,fk,opts);
}

