#ifndef FINUFFT_F_H
#define FINUFFT_F_H

#include <dataTypes.h>
#include <finufft.h>

// This defines a rather simple fortran77-style interface to FINUFFT.
// (although fortran compilers won't usually look at this C header file)

/* Note our typedefs:
   FLT = double (or float, depending on compilation precision)
   CPX = double complex (or float complex, depending on compilation precision)
   BIGINT = int64 (integer*8 in fortran)

   Make sure you call this library with matching fortran types
*/


#ifdef SINGLE
#define FINUFFT_MAKEPLAN_ finufftf_makeplan_
#define FINUFFT_SETPTS_ finufftf_setpts_
#define FINUFFT_EXEC_ finufftf_exec_
#define FINUFFT_DESTROY_ finufftf_destroy_
#define FINUFFT_DEFAULT_OPTS_ finufftf_default_opts_
#define FINUFFT1D1_ finufftf1d1_
#define FINUFFT1D1MANY_ finufftf1d1many_
#define FINUFFT1D2_ finufftf1d2_
#define FINUFFT1D2MANY_ finufftf1d2many_
#define FINUFFT1D3_ finufftf1d3_
#define FINUFFT1D3MANY_ finufftf1d3many_
#define FINUFFT2D1_ finufftf2d1_
#define FINUFFT2D1MANY_ finufftf2d1many_
#define FINUFFT2D2_ finufftf2d2_
#define FINUFFT2D2MANY_ finufftf2d2many_
#define FINUFFT2D3_ finufftf2d3_
#define FINUFFT2D3MANY_ finufftf2d3many_
#define FINUFFT3D1_ finufftf3d1_
#define FINUFFT3D1MANY_ finufftf3d1many_
#define FINUFFT3D2_ finufftf3d2_
#define FINUFFT3D2MANY_ finufftf3d2many_
#define FINUFFT3D3_ finufftf3d3_
#define FINUFFT3D3MANY_ finufftf3d3many_
/* Legacy Interfaces*/
#define FINUFFT1D1_F_ finufftf1d1_f_
#define FINUFFT1D2_F_ finufftf1d2_f_
#define FINUFFT1D3_F_ finufftf1d3_f_
#define FINUFFT2D1_F_ finufftf2d1_f_
#define FINUFFT2D2_F_ finufftf2d2_f_
#define FINUFFT2D3_F_ finufftf2d3_f_
#define FINUFFT3D1_F_ finufftf3d1_f_
#define FINUFFT3D2_F_ finufftf3d2_f_
#define FINUFFT3D3_F_ finufftf3d3_f_
#define FINUFFT2D1MANY_F_ finufftf2d1many_f_
#define FINUFFT2D2MANY_F_ finufftf2d2many_f_
#else
#define FINUFFT_MAKEPLAN_ finufft_makeplan_
#define FINUFFT_SETPTS_ finufft_setpts_
#define FINUFFT_EXEC_ finufft_exec_
#define FINUFFT_DESTROY_ finufft_destroy_
#define FINUFFT_DEFAULT_OPTS_ finufft_default_opts_
#define FINUFFT1D1_ finufft1d1_
#define FINUFFT1D1MANY_ finufft1d1many_
#define FINUFFT1D2_ finufft1d2_
#define FINUFFT1D2MANY_ finufft1d2many_
#define FINUFFT1D3_ finufft1d3_
#define FINUFFT1D3MANY_ finufft1d3many_
#define FINUFFT2D1_ finufft2d1_
#define FINUFFT2D1MANY_ finufft2d1many_
#define FINUFFT2D2_ finufft2d2_
#define FINUFFT2D2MANY_ finufft2d2many_
#define FINUFFT2D3_ finufft2d3_
#define FINUFFT2D3MANY_ finufft2d3many_
#define FINUFFT3D1_ finufft3d1_
#define FINUFFT3D1MANY_ finufft3d1many_
#define FINUFFT3D2_ finufft3d2_
#define FINUFFT3D2MANY_ finufft3d2many_
#define FINUFFT3D3_ finufft3d3_
#define FINUFFT3D3MANY_ finufft3d3many_
/* Legacy Interfaces */
#define FINUFFT1D1_F_ finufft1d1_f_
#define FINUFFT1D2_F_ finufft1d2_f_
#define FINUFFT1D3_F_ finufft1d3_f_
#define FINUFFT2D1_F_ finufft2d1_f_
#define FINUFFT2D2_F_ finufft2d2_f_
#define FINUFFT2D3_F_ finufft2d3_f_
#define FINUFFT3D1_F_ finufft3d1_f_
#define FINUFFT3D2_F_ finufft3d2_f_
#define FINUFFT3D3_F_ finufft3d3_f_
#define FINUFFT2D1MANY_F_ finufft2d1many_f_
#define FINUFFT2D2MANY_F_ finufft2d2many_f_
#endif

extern "C" {

// ---------------- the guru interface ---------------------------------
void FINUFFT_MAKEPLAN_(int *type, int *n_dims, BIGINT *n_modes, int *iflag, int *n_transf, FLT *tol, finufft_plan **plan, nufft_opts *o, int *ier);
void FINUFFT_SETPTS_(finufft_plan **plan, BIGINT *M, FLT *xj, FLT *yj, FLT *zj, BIGINT *N, FLT *s, FLT *t, FLT *u, int *ier);
void FINUFFT_EXEC_(finufft_plan **plan, CPX *weights, CPX *result, int *ier);
void FINUFFT_DESTROY_(finufft_plan **plan, int *ier);

// --------------- set default nufft_opts ----------------
void FINUFFT_DEFAULT_OPTS_(nufft_opts *o);

// -------------- simple and many-vector interfaces --------------------
// --- 1D ---
void FINUFFT1D1_(BIGINT* nj, FLT* xj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, CPX* fk, nufft_opts* o, int* ier);
void FINUFFT1D1MANY_(int* ntransf,
                 BIGINT* nj, FLT* xj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, CPX* fk, nufft_opts* o, int* ier);

void FINUFFT1D2_(BIGINT* nj, FLT* xj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, CPX* fk, nufft_opts* o, int* ier);
void FINUFFT1D2MANY_(int* ntransf,
                 BIGINT* nj, FLT* xj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, CPX* fk, nufft_opts* o, int* ier);

void FINUFFT1D3_(BIGINT* nj, FLT* x, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, CPX* f, nufft_opts* o, int* ier);
void FINUFFT1D3MANY_(int* ntransf,
                 BIGINT* nj, FLT* x, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, CPX* f, nufft_opts* o, int* ier);
// --- 2D ---
void FINUFFT2D1_(BIGINT* nj, FLT* xj, FLT* yj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, CPX* fk, nufft_opts* o, int* ier);
void FINUFFT2D1MANY_(int* ntransf,
                 BIGINT* nj, FLT* xj, FLT* yj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, CPX* fk, nufft_opts* o, int* ier);

void FINUFFT2D2_(BIGINT* nj, FLT* xj, FLT* yj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, CPX* fk, nufft_opts* o, int* ier);
void FINUFFT2D2MANY_(int* ntransf,
                 BIGINT* nj, FLT* xj, FLT* yj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, CPX* fk, nufft_opts* o, int* ier);

void FINUFFT2D3_(BIGINT* nj, FLT* x, FLT* y, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, FLT* t, CPX* f, nufft_opts* o, int* ier);
void FINUFFT2D3MANY_(int* ntransf,
                 BIGINT* nj, FLT* x, FLT* y, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, FLT* t, CPX* f, nufft_opts* o, int* ier);
// --- 3D ---
void FINUFFT3D1_(BIGINT* nj, FLT* xj, FLT* yj, FLT* zj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, BIGINT* mu, CPX* fk, nufft_opts* o, int* ier);
void FINUFFT3D1MANY_(int* ntransf,
                 BIGINT* nj, FLT* xj, FLT* yj, FLT* zj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, BIGINT* mu, CPX* fk, nufft_opts* o, int* ier);

void FINUFFT3D2_(BIGINT* nj, FLT* xj, FLT* yj, FLT* zj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, BIGINT* mu, CPX* fk, nufft_opts* o, int* ier);
void FINUFFT3D2MANY_(int* ntransf,
                 BIGINT* nj, FLT* xj, FLT* yj, FLT* zj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, BIGINT* mu, CPX* fk, nufft_opts* o, int* ier);

void FINUFFT3D3_(BIGINT* nj, FLT* x, FLT* y, FLT* z, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, FLT* t, FLT* u, CPX* f, nufft_opts* o, int* ier);
void FINUFFT3D3MANY_(int* ntransf,
                 BIGINT* nj, FLT* x, FLT* y, FLT* z, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, FLT* t, FLT* u, CPX* f, nufft_opts* o, int* ier);
// -------------- end of simple and many-vector interfaces -------------

}

#endif
