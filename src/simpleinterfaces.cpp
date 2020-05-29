#include <finufft.h>
#include <dataTypes.h>
#include <defs.h>

#include <stdio.h>
#include <iostream>
#include <iomanip>

// ---------------------------------------------------------------------------
// The 18 simple interfaces (= 3 dims * 3 types * {singlecall,many}) to FINUFFT.
// As of v1.2 these simply invoke the guru interface, through a helper layer.
// See ../docs/usage.rst or http://finufft.readthedocs.io for documentation
// all routines here.
// Authors: Andrea Malleo and Alex Barnett, 2019-2020.
// ---------------------------------------------------------------------------


// Helper layer ...........................................................

int invokeGuruInterface(int n_dims, int type, int n_transf, BIGINT nj, FLT* xj,
                        FLT *yj, FLT *zj, CPX* cj,int iflag, FLT eps,
                        BIGINT *n_modes, BIGINT nk, FLT *s, FLT *t,  FLT *u,
                        CPX* fk, nufft_opts *popts)
// Helper layer between simple interfaces (with opts) and the guru functions.
// Author: Andrea Malleo, 2019.
{
  finufft_plan plan;
  int ier = finufft_makeplan(type, n_dims, n_modes, iflag, n_transf, eps,
                             &plan, popts);  // popts (ptr to opts) can be NULL
  if (ier){
    fprintf(stderr, "finufft invokeGuru: plan error (ier=%d)!\n", ier);
    return ier;
  }
  
  ier = finufft_setpts(&plan, nj, xj, yj, zj, nk, s, t, u);
  if (ier){
    fprintf(stderr,"finufft invokeGuru: setpts error (ier=%d)!\n", ier);
    return ier;
  }

  ier = finufft_exec(&plan, cj, fk);
  if (ier){
    fprintf(stderr,"finufft invokeGuru: exec error (ier=%d)!\n", ier);
    return ier;
  }

  finufft_destroy(&plan);  
  return 0;
}



// Dimension 1111111111111111111111111111111111111111111111111111111111111111

int finufft1d1(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
	       CPX* fk, nufft_opts *opts)
//  Type-1 1D complex nonuniform FFT. See ../docs/usage.rst
{
  BIGINT n_modes[]={ms,1,1};
  int n_dims = 1;
  int n_transf = 1;
  int type = 1;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, NULL, NULL, cj,
			 iflag, eps, n_modes, 0, NULL, NULL, NULL, fk, opts);
   return ier;
}

int finufft1d1many(int n_transf, BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,
                   BIGINT ms, CPX* fk, nufft_opts *opts)
// Type-1 1D complex nonuniform FFT for many vectors. See ../docs/usage.rst
{
  BIGINT n_modes[]={ms,1,1};
  int n_dims = 1;
  int type = 1;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, NULL, NULL, cj,
		      iflag, eps, n_modes, 0, NULL, NULL, NULL, fk, opts);
  return ier;
}

int finufft1d2(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
	       CPX* fk, nufft_opts *opts)
//  Type-2 1D complex nonuniform FFT. See ../docs/usage.rst
{
  BIGINT n_modes[]={ms,1,1};
  int n_dims = 1;
  int n_transf = 1;
  int type = 2;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, NULL, NULL, cj,
			  iflag, eps, n_modes, 0, NULL, NULL, NULL, fk, opts);
  return ier;
}

int finufft1d2many(int n_transf, BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
	       CPX* fk, nufft_opts *opts)
//  Type-2 1D complex nonuniform FFT, many vectors. See ../docs/usage.rst
{
  BIGINT n_modes[]={ms,1,1};
  int n_dims = 1;
  int type = 2;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, NULL, NULL, cj,
		      	iflag, eps, n_modes, 0, NULL, NULL, NULL, fk, opts);
  return ier;
}

int finufft1d3(BIGINT nj,FLT* xj,CPX* cj,int iflag, FLT eps, BIGINT nk, FLT* s, CPX* fk, nufft_opts *opts)
// Type-3 1D complex nonuniform FFT. See ../docs/usage.rst
{
  int n_dims = 1;
  int n_transf = 1;
  int type = 3;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, NULL, NULL, cj,
				iflag, eps, NULL, nk, s, NULL, NULL, fk, opts);
  return ier;
}

int finufft1d3many(int n_transf, BIGINT nj,FLT* xj,CPX* cj,int iflag, FLT eps, BIGINT nk, FLT* s, CPX* fk, nufft_opts *opts)
  // Type-3 1D complex nonuniform FFT, many vectors. See ../docs/usage.rst
{
  int n_dims = 1;
  int type = 3;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, NULL, NULL, cj,
				iflag, eps, NULL, nk, s, NULL, NULL, fk, opts);
  return ier;
}


// Dimension 22222222222222222222222222222222222222222222222222222222222222222

int finufft2d1(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,
	       FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts* opts)
//  Type-1 2D complex nonuniform FFT. See ../docs/usage.rst
{
  BIGINT n_modes[]={ms,mt,1};
  int n_dims = 2;
  int n_transf = 1;
  int type = 1;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, NULL, cj,
                          iflag, eps, n_modes, 0, NULL, NULL, NULL,fk, opts);
  return ier; 
}

int finufft2d1many(int n_transf, BIGINT nj, FLT* xj, FLT *yj, CPX* c,
		   int iflag, FLT eps, BIGINT ms, BIGINT mt, CPX* fk,
		   nufft_opts *opts)
//  Type-1 2D complex nonuniform FFT, many vectors. See ../docs/usage.rst
{
  BIGINT n_modes[]={ms,mt,1};
  int n_dims = 2;
  int type = 1;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj,NULL, c,
                        iflag, eps, n_modes, 0, NULL, NULL, NULL, fk, opts);
  return ier; 
}

int finufft2d2(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, CPX* fk, nufft_opts *opts)
//  Type-2 2D complex nonuniform FFT.  See ../docs/usage.rst
{  
  BIGINT n_modes[]={ms,mt,1};
  int n_dims = 2;
  int n_transf = 1;
  int type = 2;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, NULL, cj, iflag,
				eps, n_modes, 0, NULL, NULL, NULL, fk, opts);
  return ier;
}

int finufft2d2many(int n_transf, BIGINT nj, FLT* xj, FLT *yj, CPX* c, int iflag,
		   FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts *opts)
//  Type-2 2D complex nonuniform FFT, many vectors.  See ../docs/usage.rst
{
  BIGINT n_modes[]={ms,mt,1};
  int n_dims = 2;
  int type = 2;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, NULL, c, iflag,
				eps, n_modes, 0, NULL, NULL, NULL, fk, opts);
  return ier; 
}

int finufft2d3(BIGINT nj,FLT* xj,FLT* yj,CPX* cj,int iflag, FLT eps, BIGINT nk, FLT* s, FLT *t, CPX* fk, nufft_opts *opts)
// Type-3 2D complex nonuniform FFT.  See ../docs/usage.rst
{
  int n_dims = 2;
  int type = 3;
  int n_transf = 1;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, NULL, cj,iflag, eps, NULL, nk, s,t,NULL, fk, opts);
  return ier;  
}

int finufft2d3many(int n_transf, BIGINT nj,FLT* xj,FLT* yj,CPX* cj,int iflag, FLT eps, BIGINT nk, FLT* s, FLT *t, CPX* fk, nufft_opts *opts)
// Type-3 2D complex nonuniform FFT, many vectors.  See ../docs/usage.rst
{
  int n_dims = 2;
  int type = 3;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, NULL, cj,iflag, eps, NULL, nk, s,t,NULL, fk, opts);
  return ier;
}



// Dimension 3333333333333333333333333333333333333333333333333333333333333333

int finufft3d1(BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,
	       FLT eps, BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk,
	       nufft_opts *opts)
//  Type-1 3D complex nonuniform FFT.   See ../docs/usage.rst
{
  BIGINT n_modes[]={ms,mt,mu};
  int n_dims = 3;
  int n_transf = 1;
  int type = 1;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, zj, cj, iflag,
				eps, n_modes, 0, NULL, NULL, NULL, fk, opts);
  return ier;
}


int finufft3d1many(int n_transf, BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,
                   int iflag, FLT eps, BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk,
                   nufft_opts *opts)
// Type-1 3D complex nonuniform FFT, many vectors.  See ../docs/usage.rst
{
  BIGINT n_modes[]={ms,mt,mu};
  int n_dims = 3;
  int type = 1;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, zj, cj, iflag,
				eps, n_modes, 0,  NULL, NULL, NULL, fk, opts);
  return ier;
}

int finufft3d2(BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,
	       int iflag,FLT eps, BIGINT ms, BIGINT mt, BIGINT mu,
	       CPX* fk, nufft_opts *opts)
// Type-2 3D complex nonuniform FFT.   See ../docs/usage.rst
{
  BIGINT n_modes[]={ms,mt,mu};
  int n_dims = 3;
  int n_transf = 1;
  int type = 2;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, zj, cj, iflag,
				eps, n_modes, 0, NULL, NULL, NULL, fk, opts);
  return ier;
}

int finufft3d2many(int n_transf, BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,
	       int iflag,FLT eps, BIGINT ms, BIGINT mt, BIGINT mu,
	       CPX* fk, nufft_opts *opts)
// Type-2 3D complex nonuniform FFT, many vectors.   See ../docs/usage.rst
{
  BIGINT n_modes[]={ms,mt,mu};
  n_modes[0] = ms;
  n_modes[1] = mt;
  n_modes[2] = mu;
  int n_dims = 3;
  int type = 2;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, zj, cj, iflag,
				eps, n_modes, 0, NULL, NULL, NULL, fk, opts);
  return ier;
}

int finufft3d3(BIGINT nj,FLT* xj,FLT* yj,FLT *zj, CPX* cj,
	       int iflag, FLT eps, BIGINT nk, FLT* s, FLT *t,
	       FLT *u, CPX* fk, nufft_opts *opts)
//  Type-3 3D complex nonuniform FFT.   See ../docs/usage.rst
{
  int n_dims = 3;
  int n_transf = 1;
  int type = 3;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, zj, cj, iflag,
				eps, NULL, nk, s ,t ,u, fk, opts);
  return ier;
}

int finufft3d3many(int n_transf, BIGINT nj,FLT* xj,FLT* yj,FLT *zj, CPX* cj,
	       int iflag, FLT eps, BIGINT nk, FLT* s, FLT *t,
	       FLT *u, CPX* fk, nufft_opts *opts)
//  Type-3 3D complex nonuniform FFT, many vectors.   See ../docs/usage.rst
{
  int n_dims = 3;
  int type = 3;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, zj, cj, iflag,
				eps, NULL, nk, s ,t ,u, fk, opts);
  return ier;
}
