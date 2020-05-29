#ifndef FINUFFT_F_H
#define FINUFFT_F_H

#include <dataTypes.h>
#include <finufft.h>

// This defines a rather simple fortran77-style interface to FINUFFT.

/* note our typedefs:
   FLT = double (or float, depending on compilation precision)
   CPX = double complex (or float complex, depending on compilation precision)
   BIGINT = int64 (integer*8 in fortran)

   Make sure you call this library with appropriate fortran sizes
*/

extern "C" {

// ---------------- the guru interface ---------------------------------
void finufft_makeplan_(int *type, int *n_dims, BIGINT *n_modes, int *iflag, int *n_transf, FLT *tol, finufft_plan **plan, nufft_opts **o, int *ier);
void finufft_setpts_(finufft_plan **plan, BIGINT *M, FLT *xj, FLT *yj, FLT *zj, BIGINT *N, FLT *s, FLT *t, FLT *u, int *ier);
void finufft_exec_(finufft_plan **plan, CPX *weights, CPX *result, int *ier);
void finufft_destroy_(finufft_plan **plan, int *ier);

// --------------- create nufft_opts and set attributes ----------------
void finufft_default_opts_(nufft_opts **o);
void set_debug_(nufft_opts **o, int *debug);
void set_spread_debug_(nufft_opts **o, int *spread_debug);
void set_spread_kerevalmeth_(nufft_opts **o, int *spread_kerevalmeth);
void set_spread_kerpad_(nufft_opts **o, int *spread_kerpad);
void set_chkbnds_(nufft_opts **o, int *chkbnds);
void set_fftw_(nufft_opts **o, int *fftw);
void set_modeord_(nufft_opts **o, int *modeord);
void set_upsampfac_(nufft_opts **o, FLT *upsampfac);
void set_spread_thread_(nufft_opts **o, int *spread_thread);
void set_maxbatchsize_(nufft_opts **o, int *maxbatchsize);

// -------------- simple and many-vector interfaces --------------------
// --- 1D ---
void finufft1d1_(BIGINT* nj, FLT* xj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, CPX* fk, nufft_opts** o, int* ier);
void finufft1d1many_(int* ntransf,
                 BIGINT* nj, FLT* xj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, CPX* fk, nufft_opts** o, int* ier);

void finufft1d2_(BIGINT* nj, FLT* xj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, CPX* fk, nufft_opts** o, int* ier);
void finufft1d2many_(int* ntransf,
                 BIGINT* nj, FLT* xj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, CPX* fk, nufft_opts** o, int* ier);

void finufft1d3_(BIGINT* nj, FLT* x, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, CPX* f, nufft_opts** o, int* ier);
void finufft1d3many_(int* ntransf,
                 BIGINT* nj, FLT* x, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, CPX* f, nufft_opts** o, int* ier);
// --- 2D ---
void finufft2d1_(BIGINT* nj, FLT* xj, FLT* yj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, CPX* fk, nufft_opts** o, int* ier);
void finufft2d1many_(int* ntransf,
                 BIGINT* nj, FLT* xj, FLT* yj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, CPX* fk, nufft_opts** o, int* ier);

void finufft2d2_(BIGINT* nj, FLT* xj, FLT* yj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, CPX* fk, nufft_opts** o, int* ier);
void finufft2d2many_(int* ntransf,
                 BIGINT* nj, FLT* xj, FLT* yj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, CPX* fk, nufft_opts** o, int* ier);

void finufft2d3_(BIGINT* nj, FLT* x, FLT* y, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, FLT* t, CPX* f, nufft_opts** o, int* ier);
void finufft2d3many_(int* ntransf,
                 BIGINT* nj, FLT* x, FLT* y, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, FLT* t, CPX* f, nufft_opts** o, int* ier);
// --- 3D ---
void finufft3d1_(BIGINT* nj, FLT* xj, FLT* yj, FLT* zj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, BIGINT* mu, CPX* fk, nufft_opts** o, int* ier);
void finufft3d1many_(int* ntransf,
                 BIGINT* nj, FLT* xj, FLT* yj, FLT* zj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, BIGINT* mu, CPX* fk, nufft_opts** o, int* ier);

void finufft3d2_(BIGINT* nj, FLT* xj, FLT* yj, FLT* zj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, BIGINT* mu, CPX* fk, nufft_opts** o, int* ier);
void finufft3d2many_(int* ntransf,
                 BIGINT* nj, FLT* xj, FLT* yj, FLT* zj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, BIGINT* mu, CPX* fk, nufft_opts** o, int* ier);

void finufft3d3_(BIGINT* nj, FLT* x, FLT* y, FLT* z, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, FLT* t, FLT* u, CPX* f, nufft_opts** o, int* ier);
void finufft3d3many_(int* ntransf,
                 BIGINT* nj, FLT* x, FLT* y, FLT* z, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, FLT* t, FLT* u, CPX* f, nufft_opts** o, int* ier);
// -------------- end of simple and many-vector interfaces -------------

}

#endif
