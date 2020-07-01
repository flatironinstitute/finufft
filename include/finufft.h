// minimum definitions needed for interface to FINUFFT library, from C++ or C

#ifndef FINUFFT_H
#define FINUFFT_H

// just what's needed to describe the headers for what finufft provides
#include <dataTypes.h>
#include <nufft_opts.h>
#include <finufft_plan.h>

// interfaces finufft.cpp provides to the outside world only (all macroed)...
#ifdef SINGLE
#define FINUFFT_DEFAULT_OPTS finufftf_default_opts
#define FINUFFT_MAKEPLAN finufftf_makeplan
#define FINUFFT_SETPTS finufftf_setpts
#define FINUFFT_EXEC finufftf_exec
#define FINUFFT_DESTROY finufftf_destroy
#define FINUFFT1D1 finufftf1d1
#define FINUFFT1D1MANY finufftf1d1many
#define FINUFFT1D2 finufftf1d2
#define FINUFFT1D2MANY finufftf1d2many
#define FINUFFT1D3 finufftf1d3
#define FINUFFT1D3MANY finufftf1d3many
#define FINUFFT2D1 finufftf2d1
#define FINUFFT2D1MANY finufftf2d1many
#define FINUFFT2D2 finufftf2d2
#define FINUFFT2D2MANY finufftf2d2many
#define FINUFFT2D3 finufftf2d3
#define FINUFFT2D3MANY finufftf2d3many
#define FINUFFT3D1 finufftf3d1
#define FINUFFT3D1MANY finufftf3d1many
#define FINUFFT3D2 finufftf3d2
#define FINUFFT3D2MANY finufftf3d2many
#define FINUFFT3D3 finufftf3d3
#define FINUFFT3D3MANY finufftf3d3many
#else
#define FINUFFT_DEFAULT_OPTS finufft_default_opts
#define FINUFFT_MAKEPLAN finufft_makeplan
#define FINUFFT_SETPTS finufft_setpts
#define FINUFFT_EXEC finufft_exec
#define FINUFFT_DESTROY finufft_destroy
#define FINUFFT1D1 finufft1d1
#define FINUFFT1D1MANY finufft1d1many
#define FINUFFT1D2 finufft1d2
#define FINUFFT1D2MANY finufft1d2many
#define FINUFFT1D3 finufft1d3
#define FINUFFT1D3MANY finufft1d3many
#define FINUFFT2D1 finufft2d1
#define FINUFFT2D1MANY finufft2d1many
#define FINUFFT2D2 finufft2d2
#define FINUFFT2D2MANY finufft2d2many
#define FINUFFT2D3 finufft2d3
#define FINUFFT2D3MANY finufft2d3many
#define FINUFFT3D1 finufft3d1
#define FINUFFT3D1MANY finufft3d1many
#define FINUFFT3D2 finufft3d2
#define FINUFFT3D2MANY finufft3d2many
#define FINUFFT3D3 finufft3d3
#define FINUFFT3D3MANY finufft3d3many
#endif

#ifdef __cplusplus
extern "C"
{
#endif

// ------------------ the guru interface ------------------------------------
  
void FINUFFT_DEFAULT_OPTS(nufft_opts *o);
int FINUFFT_MAKEPLAN(int type, int dim, BIGINT* n_modes, int iflag, int n_transf, FLT tol, finufft_plan* plan, nufft_opts* o);
int FINUFFT_SETPTS(finufft_plan* plan , BIGINT M, FLT *xj, FLT *yj, FLT *zj, BIGINT N, FLT *s, FLT *t, FLT *u); 
int FINUFFT_EXEC(finufft_plan* plan, CPX* weights, CPX* result);
int FINUFFT_DESTROY(finufft_plan* plan);


// ----------------- the 18 simple interfaces -------------------------------
// (source is in simpleinterfaces.cpp rather than finufft.cpp)
  
int FINUFFT1D1(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
	       CPX* fk, nufft_opts *opts);
int FINUFFT1D1MANY(int ntransf, BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
	       CPX* fk, nufft_opts *opts);

int FINUFFT1D2(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
	       CPX* fk, nufft_opts *opts);
int FINUFFT1D2MANY(int ntransf, BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
	       CPX* fk, nufft_opts *opts);
int FINUFFT1D3(BIGINT nj,FLT* x,CPX* c,int iflag,FLT eps,BIGINT nk, FLT* s, CPX* f, nufft_opts *opts);
int FINUFFT1D3MANY(int ntransf, BIGINT nj,FLT* x,CPX* c,int iflag,FLT eps,BIGINT nk, FLT* s, CPX* f, nufft_opts *opts);
int FINUFFT2D1(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, CPX* fk, nufft_opts *opts);
int FINUFFT2D1MANY(int ndata, BIGINT nj, FLT* xj, FLT *yj, CPX* c, int iflag,
                   FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts *opts);
int FINUFFT2D2(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, CPX* fk, nufft_opts *opts);
int FINUFFT2D2MANY(int ndata, BIGINT nj, FLT* xj, FLT *yj, CPX* c, int iflag,
                   FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts *opts);
int FINUFFT2D3(BIGINT nj,FLT* x,FLT *y,CPX* cj,int iflag,FLT eps,BIGINT nk, FLT* s, FLT* t, CPX* fk, nufft_opts *opts);

int FINUFFT2D3MANY(int ntransf, BIGINT nj,FLT* x,FLT *y,CPX* cj,int iflag,FLT eps,BIGINT nk, FLT* s, FLT* t, CPX* fk, nufft_opts *opts);

int FINUFFT3D1(BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk, nufft_opts *opts);
int FINUFFT3D1MANY(int ntransfs, BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk, nufft_opts *opts);

int FINUFFT3D2(BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk, nufft_opts *opts);
int FINUFFT3D2MANY(int ntransf, BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk, nufft_opts *opts);
int FINUFFT3D3(BIGINT nj,FLT* x,FLT *y,FLT *z, CPX* cj,int iflag,
	       FLT eps,BIGINT nk,FLT* s, FLT* t, FLT *u,
	       CPX* fk, nufft_opts *opts);
int FINUFFT3D3MANY(int ntransf, BIGINT nj,FLT* x,FLT *y,FLT *z, CPX* cj,int iflag,
	       FLT eps,BIGINT nk,FLT* s, FLT* t, FLT *u,
	       CPX* fk, nufft_opts *opts);

  
#ifdef __cplusplus
}
#endif

#endif   // FINUFFT_H
