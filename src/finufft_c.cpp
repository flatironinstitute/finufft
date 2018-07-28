#include "utils.h"

extern "C" {
#include "finufft_c.h"
}

#include "finufft.h"

//#include <ccomplex>   // C++ complex which includes C-type complex

// wrappers for calling FINUFFT in C complex style, with default opts for now.
// The interface is the same as the C++ library but without the last opts arg.
// integer*4 for the array size arguments for now.
//
// The correct way to interface complex type from C to C++ is confusing, as
// is apparent by the comments in this file.

// Eg:   int finufft1d1_c(int nj,double* xj,dcomplex* cj,int iflag, double eps,int ms, dcomplex* fk)  failed below.

// Barnett 3/10/17


void finufft_default_c_opts(nufft_c_opts *o)
// user needs to see this to set up default C-struct opts
{
  nufft_opts opts;
  finufft_default_opts(opts);         // insert the C++ defaults
  o->debug = opts.debug;              // copy stuff over - why?
  o->spread_debug = opts.spread_debug;   // ...annoying & hard to maintain
  o->spread_sort = opts.spread_sort;
  o->spread_kerevalmeth = opts.spread_kerevalmeth;
  o->spread_kerpad = opts.spread_kerpad;
  o->chkbnds = opts.chkbnds;
  o->fftw = opts.fftw;
  o->modeord = opts.modeord;
  o->upsampfac = opts.upsampfac;
}

void transfer_opts(nufft_opts &opts,nufft_c_opts copts)
// internal use only: copies C-struct opts to C++ struct for calling C++
{
  opts.debug = copts.debug;
  opts.spread_debug = copts.spread_debug;
  opts.spread_sort = copts.spread_sort;
  opts.spread_kerevalmeth = copts.spread_kerevalmeth;
  opts.spread_kerpad = copts.spread_kerpad;
  opts.chkbnds = copts.chkbnds;
  opts.fftw = copts.fftw;
  opts.modeord = copts.modeord;
  opts.upsampfac = copts.upsampfac;
}

// interfaces for the C user:

int finufft1d1_c(int nj,FLT* xj,FLT _Complex* cj,int iflag, FLT eps,int ms, FLT _Complex* fk, struct nufft_c_opts copts)
{
  nufft_opts opts;
  transfer_opts(opts,copts);
  return finufft1d1((BIGINT)nj,xj,(CPX *)cj,iflag,eps,(BIGINT)ms,(CPX *)fk,opts);
}

int finufft1d2_c(int nj,FLT* xj,FLT _Complex* cj,int iflag, FLT eps,int ms,
	       FLT _Complex* fk, nufft_c_opts copts)
{
  nufft_opts opts;
  transfer_opts(opts,copts);
  return finufft1d2(BIGINT(nj),xj,(CPX*) cj,iflag,eps,(BIGINT)ms,
		    (CPX* )fk, opts);
}

int finufft1d3_c(int nj,FLT* x,FLT _Complex* c,int iflag,FLT eps,int nk, FLT* s, FLT _Complex* f, nufft_c_opts copts)
{
  nufft_opts opts;
  transfer_opts(opts,copts);
  return finufft1d3((BIGINT)nj,x,(CPX*)c,iflag,eps,(BIGINT)nk, s, (CPX*) f, opts);
}

int finufft2d1_c(int nj,FLT* xj,FLT* yj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt, FLT _Complex* fk, nufft_c_opts copts)
{
  nufft_opts opts;
  transfer_opts(opts,copts);
  return finufft2d1((BIGINT)nj,xj,yj,(CPX *)cj,iflag,eps,(BIGINT)ms,(BIGINT)mt,(CPX *)fk,opts);
}

int finufft2d1many_c(int ndata,int nj,FLT* xj,FLT* yj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt, FLT _Complex* fk, nufft_c_opts copts)
{
  nufft_opts opts;
  transfer_opts(opts,copts);
  return finufft2d1many(ndata,(BIGINT)nj,xj,yj,(CPX *)cj,iflag,eps,(BIGINT)ms,(BIGINT)mt,(CPX *)fk,opts);
}

int finufft2d2_c(int nj,FLT* xj,FLT *yj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt, FLT _Complex* fk, nufft_c_opts copts)
{
  nufft_opts opts;
  transfer_opts(opts,copts);
  return finufft2d2(BIGINT(nj),xj,yj,(CPX*) cj,iflag,eps,(BIGINT)ms,(BIGINT)mt,
		    (CPX* )fk, opts);
}

int finufft2d2many_c(int ndata,int nj,FLT* xj,FLT *yj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt, FLT _Complex* fk, nufft_c_opts copts)
{
  nufft_opts opts;
  transfer_opts(opts,copts);
  return finufft2d2many(ndata,BIGINT(nj),xj,yj,(CPX*) cj,iflag,eps,(BIGINT)ms,
			(BIGINT)mt,(CPX* )fk, opts);
}

int finufft2d3_c(int nj,FLT* x,FLT *y,FLT _Complex* c,int iflag,FLT eps,int nk, FLT* s, FLT *t,FLT _Complex* f, nufft_c_opts copts)
{
  nufft_opts opts;
  transfer_opts(opts,copts);
  return finufft2d3((BIGINT)nj,x,y,(CPX*)c,iflag,eps,(BIGINT)nk, s, t,(CPX*) f, opts);
}

int finufft3d1_c(int nj,FLT* xj,FLT* yj,FLT *zj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt, int mu,FLT _Complex* fk, nufft_c_opts copts)
{
  nufft_opts opts;
  transfer_opts(opts,copts);
  return finufft3d1((BIGINT)nj,xj,yj,zj,(CPX *)cj,iflag,eps,(BIGINT)ms,(BIGINT)mt,(BIGINT)mu,(CPX *)fk,opts);
}

int finufft3d2_c(int nj,FLT* xj,FLT *yj,FLT *zj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt, int mu, FLT _Complex* fk, nufft_c_opts copts)
{
  nufft_opts opts;
  transfer_opts(opts,copts);
  return finufft3d2(BIGINT(nj),xj,yj,zj,(CPX*) cj,iflag,eps,(BIGINT)ms,(BIGINT)mt,(BIGINT)mu, (CPX* )fk, opts);
}

int finufft3d3_c(int nj,FLT* x,FLT *y,FLT *z,FLT _Complex* c,int iflag,FLT eps,int nk, FLT* s, FLT *t,FLT *u,FLT _Complex* f, nufft_c_opts copts)
{
  nufft_opts opts;
  transfer_opts(opts,copts);
  return finufft3d3((BIGINT)nj,x,y,z,(CPX*)c,iflag,eps,(BIGINT)nk, s, t,u,(CPX*) f, opts);
}


