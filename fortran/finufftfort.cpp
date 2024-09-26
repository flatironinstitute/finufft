/* C++ layer for calling FINUFFT from fortran, in f77 style + derived type
   for the finufft_opts C-struct. The ptr to finufft_plan is passed as an "opaque"
   pointer (as in FFTW3 legacy fortran interface).
   Note the trailing underscore name-mangle which is not needed from fortran.

   Note our typedefs:
   FLT = double (or float, depending on compilation precision)
   CPX = double complex (or float complex, depending on compilation precision)
   BIGINT = int64 (integer*8 in fortran)

   Make sure you call this library with matching fortran types

   For a demo see: examples/simple1d1.f

   Barnett 2/17/17. Single prec 4/5/17. Libin Lu & Alex Barnett, May 2020.
   Garrett Wright dual-prec 6/28/20.
   Barnett safe-header, cleaned up, 6/7/22.
*/

// public header
#include <finufft.h>
#include <finufft/finufft_core.h>

#ifdef __cplusplus
extern "C" {
#endif

// --------------------- guru interface from fortran ------------------------
void finufft_makeplan_(int *type, int *n_dims, BIGINT *n_modes, int *iflag, int *n_transf,
                       double *tol, finufft_plan *plan, finufft_opts *o, int *ier) {
  if (!plan)
    fprintf(stderr,
            "%s fortran: plan must be allocated as at least the size of a C pointer "
            "(usually 8 bytes)!\n",
            __func__);
  else {
    // pass o whether it's a NULL or pointer to a fortran-allocated finufft_opts:
    *ier = finufft_makeplan(*type, *n_dims, n_modes, *iflag, *n_transf, *tol, plan, o);
  }
}

void finufft_setpts_(finufft_plan *plan, BIGINT *M, double *xj, double *yj, double *zj,
                     BIGINT *nk, double *s, double *t, double *u, int *ier) {
  if (!*plan) {
    fprintf(stderr, "%s fortran: finufft_plan unallocated!", __func__);
    return;
  }
  int nk_safe = 0; // catches the case where user passes NULL in
  if (nk) nk_safe = int(*nk);
  *ier = finufft_setpts(*plan, *M, xj, yj, zj, nk_safe, s, t, u);
}

void finufft_execute_(finufft_plan *plan, std::complex<double> *weights,
                      std::complex<double> *result, int *ier) {
  if (!plan)
    fprintf(stderr, "%s fortran: finufft_plan unallocated!", __func__);
  else
    *ier = finufft_execute(*plan, weights, result);
}

void finufft_destroy_(finufft_plan *plan, int *ier) {
  if (!plan)
    fprintf(stderr, "%s fortran: finufft_plan unallocated!", __func__);
  else
    *ier = finufft_destroy(*plan);
}

// ------------ use FINUFFT to set the default options ---------------------
// (Note the finufft_opts is created in f90-style derived types, not here)
void finufft_default_opts_(finufft_opts *o) {
  if (!o)
    fprintf(stderr, "%s fortran: opts must be allocated!\n", __func__);
  else
    // o is a ptr to already-allocated fortran finufft_opts derived type...
    finufft_default_opts(o);
}

// -------------- simple and many-vector interfaces --------------------
// --- 1D ---
void finufft1d1_(BIGINT *nj, double *xj, std::complex<double> *cj, int *iflag,
                 double *eps, BIGINT *ms, std::complex<double> *fk, finufft_opts *o,
                 int *ier) {
  *ier = finufft1d1(*nj, xj, cj, *iflag, *eps, *ms, fk, o);
}

void finufft1d1many_(int *ntransf, BIGINT *nj, double *xj, std::complex<double> *cj,
                     int *iflag, double *eps, BIGINT *ms, std::complex<double> *fk,
                     finufft_opts *o, int *ier) {
  *ier = finufft1d1many(*ntransf, *nj, xj, cj, *iflag, *eps, *ms, fk, o);
}

void finufft1d2_(BIGINT *nj, double *xj, std::complex<double> *cj, int *iflag,
                 double *eps, BIGINT *ms, std::complex<double> *fk, finufft_opts *o,
                 int *ier) {
  *ier = finufft1d2(*nj, xj, cj, *iflag, *eps, *ms, fk, o);
}

void finufft1d2many_(int *ntransf, BIGINT *nj, double *xj, std::complex<double> *cj,
                     int *iflag, double *eps, BIGINT *ms, std::complex<double> *fk,
                     finufft_opts *o, int *ier) {
  *ier = finufft1d2many(*ntransf, *nj, xj, cj, *iflag, *eps, *ms, fk, o);
}

void finufft1d3_(BIGINT *nj, double *x, std::complex<double> *c, int *iflag, double *eps,
                 BIGINT *nk, double *s, std::complex<double> *f, finufft_opts *o,
                 int *ier) {
  *ier = finufft1d3(*nj, x, c, *iflag, *eps, *nk, s, f, o);
}

void finufft1d3many_(int *ntransf, BIGINT *nj, double *x, std::complex<double> *c,
                     int *iflag, double *eps, BIGINT *nk, double *s,
                     std::complex<double> *f, finufft_opts *o, int *ier) {
  *ier = finufft1d3many(*ntransf, *nj, x, c, *iflag, *eps, *nk, s, f, o);
}

// --- 2D ---
void finufft2d1_(BIGINT *nj, double *xj, double *yj, std::complex<double> *cj, int *iflag,
                 double *eps, BIGINT *ms, BIGINT *mt, std::complex<double> *fk,
                 finufft_opts *o, int *ier) {
  *ier = finufft2d1(*nj, xj, yj, cj, *iflag, *eps, *ms, *mt, fk, o);
}
void finufft2d1many_(int *ntransf, BIGINT *nj, double *xj, double *yj,
                     std::complex<double> *cj, int *iflag, double *eps, BIGINT *ms,
                     BIGINT *mt, std::complex<double> *fk, finufft_opts *o, int *ier) {
  *ier = finufft2d1many(*ntransf, *nj, xj, yj, cj, *iflag, *eps, *ms, *mt, fk, o);
}

void finufft2d2_(BIGINT *nj, double *xj, double *yj, std::complex<double> *cj, int *iflag,
                 double *eps, BIGINT *ms, BIGINT *mt, std::complex<double> *fk,
                 finufft_opts *o, int *ier) {
  *ier = finufft2d2(*nj, xj, yj, cj, *iflag, *eps, *ms, *mt, fk, o);
}
void finufft2d2many_(int *ntransf, BIGINT *nj, double *xj, double *yj,
                     std::complex<double> *cj, int *iflag, double *eps, BIGINT *ms,
                     BIGINT *mt, std::complex<double> *fk, finufft_opts *o, int *ier) {
  *ier = finufft2d2many(*ntransf, *nj, xj, yj, cj, *iflag, *eps, *ms, *mt, fk, o);
}

void finufft2d3_(BIGINT *nj, double *x, double *y, std::complex<double> *c, int *iflag,
                 double *eps, BIGINT *nk, double *s, double *t, std::complex<double> *f,
                 finufft_opts *o, int *ier) {
  *ier = finufft2d3(*nj, x, y, c, *iflag, *eps, *nk, s, t, f, o);
}

void finufft2d3many_(int *ntransf, BIGINT *nj, double *x, double *y,
                     std::complex<double> *c, int *iflag, double *eps, BIGINT *nk,
                     double *s, double *t, std::complex<double> *f, finufft_opts *o,
                     int *ier) {
  *ier = finufft2d3many(*ntransf, *nj, x, y, c, *iflag, *eps, *nk, s, t, f, o);
}

// --- 3D ---
void finufft3d1_(BIGINT *nj, double *xj, double *yj, double *zj, std::complex<double> *cj,
                 int *iflag, double *eps, BIGINT *ms, BIGINT *mt, BIGINT *mu,
                 std::complex<double> *fk, finufft_opts *o, int *ier) {
  *ier = finufft3d1(*nj, xj, yj, zj, cj, *iflag, *eps, *ms, *mt, *mu, fk, o);
}

void finufft3d1many_(int *ntransf, BIGINT *nj, double *xj, double *yj, double *zj,
                     std::complex<double> *cj, int *iflag, double *eps, BIGINT *ms,
                     BIGINT *mt, BIGINT *mu, std::complex<double> *fk, finufft_opts *o,
                     int *ier) {
  *ier =
      finufft3d1many(*ntransf, *nj, xj, yj, zj, cj, *iflag, *eps, *ms, *mt, *mu, fk, o);
}

void finufft3d2_(BIGINT *nj, double *xj, double *yj, double *zj, std::complex<double> *cj,
                 int *iflag, double *eps, BIGINT *ms, BIGINT *mt, BIGINT *mu,
                 std::complex<double> *fk, finufft_opts *o, int *ier) {
  *ier = finufft3d2(*nj, xj, yj, zj, cj, *iflag, *eps, *ms, *mt, *mu, fk, o);
}

void finufft3d2many_(int *ntransf, BIGINT *nj, double *xj, double *yj, double *zj,
                     std::complex<double> *cj, int *iflag, double *eps, BIGINT *ms,
                     BIGINT *mt, BIGINT *mu, std::complex<double> *fk, finufft_opts *o,
                     int *ier) {
  *ier =
      finufft3d2many(*ntransf, *nj, xj, yj, zj, cj, *iflag, *eps, *ms, *mt, *mu, fk, o);
}

void finufft3d3_(BIGINT *nj, double *x, double *y, double *z, std::complex<double> *c,
                 int *iflag, double *eps, BIGINT *nk, double *s, double *t, double *u,
                 std::complex<double> *f, finufft_opts *o, int *ier) {
  *ier = finufft3d3(*nj, x, y, z, c, *iflag, *eps, *nk, s, t, u, f, o);
}

void finufft3d3many_(int *ntransf, BIGINT *nj, double *x, double *y, double *z,
                     std::complex<double> *c, int *iflag, double *eps, BIGINT *nk,
                     double *s, double *t, double *u, std::complex<double> *f,
                     finufft_opts *o, int *ier) {
  *ier = finufft3d3many(*ntransf, *nj, x, y, z, c, *iflag, *eps, *nk, s, t, u, f, o);
}

// --------------------- guru interface from fortran ------------------------
void finufftf_makeplan_(int *type, int *n_dims, BIGINT *n_modes, int *iflag,
                        int *n_transf, float *tol, finufftf_plan *plan, finufft_opts *o,
                        int *ier) {
  if (!plan)
    fprintf(stderr,
            "%s fortran: plan must be allocated as at least the size of a C pointer "
            "(usually 8 bytes)!\n",
            __func__);
  else {
    // pass o whether it's a NULL or pointer to a fortran-allocated finufft_opts:
    *ier = finufftf_makeplan(*type, *n_dims, n_modes, *iflag, *n_transf, *tol, plan, o);
  }
}

void finufftf_setpts_(finufftf_plan *plan, BIGINT *M, float *xj, float *yj, float *zj,
                      BIGINT *nk, float *s, float *t, float *u, int *ier) {
  if (!*plan) {
    fprintf(stderr, "%s fortran: finufft_plan unallocated!", __func__);
    return;
  }
  int nk_safe = 0; // catches the case where user passes NULL in
  if (nk) nk_safe = int(*nk);
  *ier = finufftf_setpts(*plan, *M, xj, yj, zj, nk_safe, s, t, u);
}

void finufftf_execute_(finufftf_plan *plan, std::complex<float> *weights,
                       std::complex<float> *result, int *ier) {
  if (!plan)
    fprintf(stderr, "%s fortran: finufft_plan unallocated!", __func__);
  else
    *ier = finufftf_execute(*plan, weights, result);
}

void finufftf_destroy_(finufftf_plan *plan, int *ier) {
  if (!plan)
    fprintf(stderr, "%s fortran: finufft_plan unallocated!", __func__);
  else
    *ier = finufftf_destroy(*plan);
}

// ------------ use FINUFFT to set the default options ---------------------
// (Note the finufft_opts is created in f90-style derived types, not here)
void finufftf_default_opts_(finufft_opts *o) {
  if (!o)
    fprintf(stderr, "%s fortran: opts must be allocated!\n", __func__);
  else
    // o is a ptr to already-allocated fortran finufft_opts derived type...
    finufft_default_opts(o);
}

// -------------- simple and many-vector interfaces --------------------
// --- 1D ---
void finufftf1d1_(BIGINT *nj, float *xj, std::complex<float> *cj, int *iflag, float *eps,
                  BIGINT *ms, std::complex<float> *fk, finufft_opts *o, int *ier) {
  *ier = finufftf1d1(*nj, xj, cj, *iflag, *eps, *ms, fk, o);
}

void finufftf1d1many_(int *ntransf, BIGINT *nj, float *xj, std::complex<float> *cj,
                      int *iflag, float *eps, BIGINT *ms, std::complex<float> *fk,
                      finufft_opts *o, int *ier) {
  *ier = finufftf1d1many(*ntransf, *nj, xj, cj, *iflag, *eps, *ms, fk, o);
}

void finufftf1d2_(BIGINT *nj, float *xj, std::complex<float> *cj, int *iflag, float *eps,
                  BIGINT *ms, std::complex<float> *fk, finufft_opts *o, int *ier) {
  *ier = finufftf1d2(*nj, xj, cj, *iflag, *eps, *ms, fk, o);
}

void finufftf1d2many_(int *ntransf, BIGINT *nj, float *xj, std::complex<float> *cj,
                      int *iflag, float *eps, BIGINT *ms, std::complex<float> *fk,
                      finufft_opts *o, int *ier) {
  *ier = finufftf1d2many(*ntransf, *nj, xj, cj, *iflag, *eps, *ms, fk, o);
}

void finufftf1d3_(BIGINT *nj, float *x, std::complex<float> *c, int *iflag, float *eps,
                  BIGINT *nk, float *s, std::complex<float> *f, finufft_opts *o,
                  int *ier) {
  *ier = finufftf1d3(*nj, x, c, *iflag, *eps, *nk, s, f, o);
}

void finufftf1d3many_(int *ntransf, BIGINT *nj, float *x, std::complex<float> *c,
                      int *iflag, float *eps, BIGINT *nk, float *s,
                      std::complex<float> *f, finufft_opts *o, int *ier) {
  *ier = finufftf1d3many(*ntransf, *nj, x, c, *iflag, *eps, *nk, s, f, o);
}

// --- 2D ---
void finufftf2d1_(BIGINT *nj, float *xj, float *yj, std::complex<float> *cj, int *iflag,
                  float *eps, BIGINT *ms, BIGINT *mt, std::complex<float> *fk,
                  finufft_opts *o, int *ier) {
  *ier = finufftf2d1(*nj, xj, yj, cj, *iflag, *eps, *ms, *mt, fk, o);
}
void finufftf2d1many_(int *ntransf, BIGINT *nj, float *xj, float *yj,
                      std::complex<float> *cj, int *iflag, float *eps, BIGINT *ms,
                      BIGINT *mt, std::complex<float> *fk, finufft_opts *o, int *ier) {
  *ier = finufftf2d1many(*ntransf, *nj, xj, yj, cj, *iflag, *eps, *ms, *mt, fk, o);
}

void finufftf2d2_(BIGINT *nj, float *xj, float *yj, std::complex<float> *cj, int *iflag,
                  float *eps, BIGINT *ms, BIGINT *mt, std::complex<float> *fk,
                  finufft_opts *o, int *ier) {
  *ier = finufftf2d2(*nj, xj, yj, cj, *iflag, *eps, *ms, *mt, fk, o);
}
void finufftf2d2many_(int *ntransf, BIGINT *nj, float *xj, float *yj,
                      std::complex<float> *cj, int *iflag, float *eps, BIGINT *ms,
                      BIGINT *mt, std::complex<float> *fk, finufft_opts *o, int *ier) {
  *ier = finufftf2d2many(*ntransf, *nj, xj, yj, cj, *iflag, *eps, *ms, *mt, fk, o);
}

void finufftf2d3_(BIGINT *nj, float *x, float *y, std::complex<float> *c, int *iflag,
                  float *eps, BIGINT *nk, float *s, float *t, std::complex<float> *f,
                  finufft_opts *o, int *ier) {
  *ier = finufftf2d3(*nj, x, y, c, *iflag, *eps, *nk, s, t, f, o);
}

void finufftf2d3many_(int *ntransf, BIGINT *nj, float *x, float *y,
                      std::complex<float> *c, int *iflag, float *eps, BIGINT *nk,
                      float *s, float *t, std::complex<float> *f, finufft_opts *o,
                      int *ier) {
  *ier = finufftf2d3many(*ntransf, *nj, x, y, c, *iflag, *eps, *nk, s, t, f, o);
}

// --- 3D ---
void finufftf3d1_(BIGINT *nj, float *xj, float *yj, float *zj, std::complex<float> *cj,
                  int *iflag, float *eps, BIGINT *ms, BIGINT *mt, BIGINT *mu,
                  std::complex<float> *fk, finufft_opts *o, int *ier) {
  *ier = finufftf3d1(*nj, xj, yj, zj, cj, *iflag, *eps, *ms, *mt, *mu, fk, o);
}

void finufftf3d1many_(int *ntransf, BIGINT *nj, float *xj, float *yj, float *zj,
                      std::complex<float> *cj, int *iflag, float *eps, BIGINT *ms,
                      BIGINT *mt, BIGINT *mu, std::complex<float> *fk, finufft_opts *o,
                      int *ier) {
  *ier =
      finufftf3d1many(*ntransf, *nj, xj, yj, zj, cj, *iflag, *eps, *ms, *mt, *mu, fk, o);
}

void finufftf3d2_(BIGINT *nj, float *xj, float *yj, float *zj, std::complex<float> *cj,
                  int *iflag, float *eps, BIGINT *ms, BIGINT *mt, BIGINT *mu,
                  std::complex<float> *fk, finufft_opts *o, int *ier) {
  *ier = finufftf3d2(*nj, xj, yj, zj, cj, *iflag, *eps, *ms, *mt, *mu, fk, o);
}

void finufftf3d2many_(int *ntransf, BIGINT *nj, float *xj, float *yj, float *zj,
                      std::complex<float> *cj, int *iflag, float *eps, BIGINT *ms,
                      BIGINT *mt, BIGINT *mu, std::complex<float> *fk, finufft_opts *o,
                      int *ier) {
  *ier =
      finufftf3d2many(*ntransf, *nj, xj, yj, zj, cj, *iflag, *eps, *ms, *mt, *mu, fk, o);
}

void finufftf3d3_(BIGINT *nj, float *x, float *y, float *z, std::complex<float> *c,
                  int *iflag, float *eps, BIGINT *nk, float *s, float *t, float *u,
                  std::complex<float> *f, finufft_opts *o, int *ier) {
  *ier = finufftf3d3(*nj, x, y, z, c, *iflag, *eps, *nk, s, t, u, f, o);
}

void finufftf3d3many_(int *ntransf, BIGINT *nj, float *x, float *y, float *z,
                      std::complex<float> *c, int *iflag, float *eps, BIGINT *nk,
                      float *s, float *t, float *u, std::complex<float> *f,
                      finufft_opts *o, int *ier) {
  *ier = finufftf3d3many(*ntransf, *nj, x, y, z, c, *iflag, *eps, *nk, s, t, u, f, o);
}

#ifdef __cplusplus
}
#endif
