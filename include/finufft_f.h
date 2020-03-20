#ifndef FINUFFT_F_H
#define FINUFFT_F_H

#include <dataTypes.h>
#include <finufft.h>

// This defines a rather simple fortran interface to the simple library calls.

// note FLT (= float or double) and CPX (= float complex or double complex)
// used here. Make sure you call with appropriate fortran sizes.

// All ints are int*4 for now in fortran interface, all nufft_opts defaults.
// TODO: make more flexible fortran interfaces


extern "C" {
void finufft1d1_f_(int *nj,FLT* xj,CPX* cj,int *iflag, FLT *eps,
		  int *ms, CPX* fk, int *ier);
void finufft1d2_f_(int *nj,FLT* xj,CPX* cj,int *iflag, FLT *eps,
		  int *ms, CPX* fk, int *ier);
void finufft1d3_f_(int *nj,FLT* xj,CPX* cj,int *iflag, FLT *eps,
		  int *nk, FLT* s, CPX* fk, int *ier);
void finufft2d1_f_(int *nj,FLT* xj,FLT *yj,CPX* cj,int *iflag,
		   FLT *eps, int *ms, int *mt, CPX* fk, int *ier);
void finufft2d2_f_(int *nj,FLT* xj,FLT *yj,CPX* cj,int *iflag,
		   FLT *eps, int *ms, int *mt, CPX* fk, int *ier);
void finufft2d3_f_(int *nj,FLT* xj,FLT* yj, CPX* cj,int *iflag,
		   FLT *eps, int *nk, FLT* s, FLT* t, CPX* fk,
		   int *ier);
void finufft3d1_f_(int *nj,FLT* xj,FLT *yj,FLT* zj,CPX* cj,
		   int *iflag, FLT *eps, int *ms, int *mt, int *mu,
		   CPX* fk, int *ier);
void finufft3d2_f_(int *nj,FLT* xj,FLT *yj,FLT* zj,CPX* cj,
		   int *iflag, FLT *eps, int *ms, int *mt, int *mu,
		   CPX* fk, int *ier);
void finufft3d3_f_(int *nj,FLT* xj,FLT* yj, FLT*zj, CPX* cj,
		   int *iflag, FLT *eps, int *nk, FLT* s, FLT* t,
		   FLT* u, CPX* fk, int *ier);
void finufft2d1many_f_(int *ndata, int *nj,FLT* xj,FLT *yj,CPX* cj,int *iflag,
		   FLT *eps, int *ms, int *mt, CPX* fk, int *ier);
void finufft2d2many_f_(int *ndata, int *nj,FLT* xj,FLT *yj,CPX* cj,int *iflag,
		   FLT *eps, int *ms, int *mt, CPX* fk, int *ier);
// ---------------- the guru interface ---------------------------------
void finufft_default_opts_f_(nufft_opts *o);
void finufft_makeplan_f_(int *type, int *n_dims, BIGINT *n_modes, int *iflag, int *n_transf, FLT *tol, int *blksize, finufft_plan *plan, nufft_opts *o, int *ier);
void finufft_setpts_f_(finufft_plan *plan, BIGINT *M, FLT *xj, FLT *yj, FLT *zj, BIGINT *N, FLT *s, FLT *t, FLT *u, int *ier);
void finufft_exec_f_(finufft_plan *plan, CPX *weights, CPX *result, int *ier);
void finufft_destroy_f_(finufft_plan *plan, void *o, int *ier);
// --------------- set nufft_opts attributes --------------------------
void set_debug_(nufft_opts *o, int *debug);
void set_spread_debug_(nufft_opts *o, int *spread_debug);
void set_spread_kerevalmeth_(nufft_opts *o, int *spread_kerevalmeth);
void set_spread_kerpad_(nufft_opts *o, int *spread_kerpad);
void set_chkbnds_(nufft_opts *o, int *chkbnds);
void set_fftw_(nufft_opts *o, int *fftw);
void set_modeord_(nufft_opts *o, int *modeord);
void set_upsampfac_(nufft_opts *o, FLT *upsampfac);
void set_spread_scheme_(nufft_opts *o, int *spread_scheme);
}

#endif
