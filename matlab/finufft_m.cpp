// C-style interface to FINUFFT library that is used for MWRAP interface.
// Note that no underscores can be used in the function names.
// We use doubles to rep big integers, since "long long" failed in mwrap.
// We tried reading MY_OMP_GET_MAX_THREADS() but since we're in MEX it always
// seems to be 1, so we don't use it.
// Barnett 3/24/17.
// fixed typecasting doubles to BIGINTs w/ correct rounding. 3/29/17
// double array for passing in all opts. 10/30/17

#include "../src/finufft.h"
#include <stdio.h>
#include <math.h>
#include <mex.h>

#define IROUND(x) (int)round(x)

void finufft_mex_setup()
{
  /* Forces MATLAB to properly initialize their FFTW library. */
  mexEvalString("fft(1:8);");
}

void finufft_mex_opts(nufft_opts &opts, double *mexo)
// global setup of finufft opts given MEX interface opts 6-long double array.
// Also sets multithreading.
{
  finufft_default_opts(opts);
  opts.debug = IROUND(mexo[0]);
  if (opts.debug>1) {   //  Any amount of debug>1 is pushed to spread_debug...
    opts.spread_debug=opts.debug-1;    // allows MATLAB users to see spread info
    opts.debug=1;
  }
  int nthreads = IROUND(mexo[1]);
  if (nthreads>0) MY_OMP_SET_NUM_THREADS(nthreads);
  opts.spread_sort=IROUND(mexo[2]);
  opts.fftw = !IROUND(mexo[3]) ? FFTW_ESTIMATE : FFTW_MEASURE;
  opts.modeord = IROUND(mexo[4]);      // unused by type-3
  opts.chkbnds = IROUND(mexo[5]);      // "
}


int finufft1d1m(double nj,double* xj,dcomplex* cj,int iflag,double eps,double ms, dcomplex* fk, double* mexo)
{
  nufft_opts opts;
  finufft_mex_setup();
  finufft_mex_opts(opts, mexo);
  return finufft1d1((BIGINT)(nj+0.5),xj,cj,iflag,eps,(BIGINT)(ms+0.5),fk,opts);
}

int finufft1d2m(double nj,double* xj,dcomplex* cj,int iflag,double eps,double ms, dcomplex* fk, double* mexo)
{
  nufft_opts opts;
  finufft_mex_setup();
  finufft_mex_opts(opts, mexo);
  return finufft1d2((BIGINT)(nj+0.5),xj,cj,iflag,eps,(BIGINT)(ms+0.5),fk,opts);
}

int finufft1d3m(double nj,double* xj,dcomplex* cj,int iflag,double eps,double nk, double* s, dcomplex* fk, double* mexo)
{
  nufft_opts opts;
  finufft_mex_setup();
  finufft_mex_opts(opts, mexo);
  return finufft1d3((BIGINT)(nj+0.5),xj,cj,iflag,eps,(BIGINT)(nk+0.5),s,fk,opts);
}

int finufft2d1m(double nj,double* xj,double* yj, dcomplex* cj,int iflag,double eps,double ms, double mt, dcomplex* fk, double *mexo)
{
  nufft_opts opts;
  finufft_mex_setup();
  finufft_mex_opts(opts, mexo);
  return finufft2d1((BIGINT)(nj+0.5),xj,yj,cj,iflag,eps,(BIGINT)(ms+0.5),(BIGINT)(mt+0.5),fk,opts);
}

int finufft2d2m(double nj,double* xj,double* yj,dcomplex* cj,int iflag,double eps,double ms, double mt, dcomplex* fk, double* mexo)
{
  nufft_opts opts;
  finufft_mex_setup();
  finufft_mex_opts(opts, mexo);
  return finufft2d2((BIGINT)(nj+0.5),xj,yj,cj,iflag,eps,(BIGINT)(ms+0.5),(BIGINT)(mt+0.5),fk,opts);
}

int finufft2d3m(double nj,double* xj,double *yj,dcomplex* cj,int iflag,double eps,double nk, double* s, double* t, dcomplex* fk, double* mexo)
{
  nufft_opts opts;
  finufft_mex_setup();
  finufft_mex_opts(opts, mexo);
  return finufft2d3((BIGINT)(nj+0.5),xj,yj,cj,iflag,eps,(BIGINT)(nk+0.5),s,t,fk,opts);
}

int finufft3d1m(double nj,double* xj,double* yj, double* zj, dcomplex* cj,int iflag,double eps,double ms, double mt, double mu, dcomplex* fk, double* mexo)
{
  nufft_opts opts;
  finufft_mex_setup();
  finufft_mex_opts(opts, mexo);
  return finufft3d1((BIGINT)(nj+0.5),xj,yj,zj,cj,iflag,eps,(BIGINT)(ms+0.5),(BIGINT)(mt+0.5),(BIGINT)(mu+0.5),fk,opts);
}

int finufft3d2m(double nj,double* xj,double* yj,double *zj,dcomplex* cj,int iflag,double eps,double ms, double mt, double mu, dcomplex* fk, double* mexo)
{
  nufft_opts opts;
  finufft_mex_setup();
  finufft_mex_opts(opts, mexo);
  return finufft3d2((BIGINT)(nj+0.5),xj,yj,zj,cj,iflag,eps,(BIGINT)(ms+0.5),(BIGINT)(mt+0.5),(BIGINT)(mu+0.5),fk,opts);
}

int finufft3d3m(double nj,double* xj,double *yj,double *zj,dcomplex* cj,int iflag,double eps,double nk, double* s, double* t, double *u, dcomplex* fk, double* mexo)
  {
  nufft_opts opts;
  finufft_mex_setup();
  finufft_mex_opts(opts, mexo);
  return finufft3d3((BIGINT)(nj+0.5),xj,yj,zj,cj,iflag,eps,(BIGINT)(nk+0.5),s,t,u,fk,opts);
}

