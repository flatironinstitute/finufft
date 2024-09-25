// public header
#include <finufft.h>
// private headers
#include <array>
#include <cstdio>
#include <finufft/defs.h>
using namespace std;

/* ---------------------------------------------------------------------------
   The 18 simple interfaces (= 3 dims * 3 types * {singlecall,many}) to FINUFFT.
   As of v1.2 these simply invoke the guru interface, through a helper layer.
   See ../docs/usage.rst or http://finufft.readthedocs.io for documentation
   all routines here.
   This compiles in either double or single precision (based on -DSINGLE),
   producing functions finufft?d?{many} or finufftf?1?{many} respectively.

   Authors: Andrea Malleo and Alex Barnett, 2019-2020.
   Safe namespacing, Barnett, May 2022.
   ---------------------------------------------------------------------------
*/

// Helper layer ...........................................................

namespace finufft {
namespace common {

static int invokeGuruInterface(int n_dims, int type, int n_transf, BIGINT nj, FLT *xj,
                               FLT *yj, FLT *zj, CPX *cj, int iflag, FLT eps,
                               const std::array<BIGINT, 3> &n_modes, BIGINT nk, FLT *s,
                               FLT *t, FLT *u, CPX *fk, finufft_opts *popts)
// Helper layer between simple interfaces (with opts) and the guru functions.
// Author: Andrea Malleo, 2019.
{
  FINUFFT_PLAN plan = nullptr;
  int ier = FINUFFT_MAKEPLAN(type, n_dims, n_modes.data(), iflag, n_transf, eps, &plan,
                             popts); // popts (ptr to opts) can be NULL
  if (ier > 1) {                     // since 1 (a warning) still allows proceeding...
    fprintf(stderr, "FINUFFT invokeGuru: plan error (ier=%d)!\n", ier);
    FINUFFT_DESTROY(plan);
    return ier;
  }

  int ier2 = FINUFFT_SETPTS(plan, nj, xj, yj, zj, nk, s, t, u);
  if (ier2 > 1) {
    fprintf(stderr, "FINUFFT invokeGuru: setpts error (ier=%d)!\n", ier2);
    FINUFFT_DESTROY(plan);
    return ier2;
  }

  int ier3 = FINUFFT_EXECUTE(plan, cj, fk);
  if (ier3 > 1) {
    fprintf(stderr, "FINUFFT invokeGuru: execute error (ier=%d)!\n", ier3);
    FINUFFT_DESTROY(plan);
    return ier3;
  }

  FINUFFT_DESTROY(plan);
  return max(max(ier, ier2), ier3); // in case any one gave a (positive!) warning
}

} // namespace common
} // namespace finufft

using namespace finufft::common;

// Dimension 1111111111111111111111111111111111111111111111111111111111111111

int FINUFFT1D1MANY(int n_transf, BIGINT nj, FLT *xj, CPX *cj, int iflag, FLT eps,
                   BIGINT ms, CPX *fk, finufft_opts *opts)
// Type-1 1D complex nonuniform FFT for many vectors. See ../docs/usage.rst
{
  return invokeGuruInterface(1, 1, n_transf, nj, xj, nullptr, nullptr, cj, iflag, eps,
                             {ms, 1, 1}, 0, nullptr, nullptr, nullptr, fk, opts);
}

int FINUFFT1D1(BIGINT nj, FLT *xj, CPX *cj, int iflag, FLT eps, BIGINT ms, CPX *fk,
               finufft_opts *opts)
//  Type-1 1D complex nonuniform FFT. See ../docs/usage.rst
{
  return FINUFFT1D1MANY(1, nj, xj, cj, iflag, eps, ms, fk, opts);
}

int FINUFFT1D2MANY(int n_transf, BIGINT nj, FLT *xj, CPX *cj, int iflag, FLT eps,
                   BIGINT ms, CPX *fk, finufft_opts *opts)
//  Type-2 1D complex nonuniform FFT, many vectors. See ../docs/usage.rst
{
  return invokeGuruInterface(1, 2, n_transf, nj, xj, NULL, NULL, cj, iflag, eps,
                             {ms, 1, 1}, 0, NULL, NULL, NULL, fk, opts);
}

int FINUFFT1D2(BIGINT nj, FLT *xj, CPX *cj, int iflag, FLT eps, BIGINT ms, CPX *fk,
               finufft_opts *opts)
//  Type-2 1D complex nonuniform FFT. See ../docs/usage.rst
{
  return FINUFFT1D2MANY(1, nj, xj, cj, iflag, eps, ms, fk, opts);
}

int FINUFFT1D3MANY(int n_transf, BIGINT nj, FLT *xj, CPX *cj, int iflag, FLT eps,
                   BIGINT nk, FLT *s, CPX *fk, finufft_opts *opts)
// Type-3 1D complex nonuniform FFT, many vectors. See ../docs/usage.rst
{
  return invokeGuruInterface(1, 3, n_transf, nj, xj, NULL, NULL, cj, iflag, eps,
                             {0, 0, 0}, nk, s, NULL, NULL, fk, opts);
}
int FINUFFT1D3(BIGINT nj, FLT *xj, CPX *cj, int iflag, FLT eps, BIGINT nk, FLT *s,
               CPX *fk, finufft_opts *opts)
// Type-3 1D complex nonuniform FFT. See ../docs/usage.rst
{
  return FINUFFT1D3MANY(1, nj, xj, cj, iflag, eps, nk, s, fk, opts);
}

// Dimension 22222222222222222222222222222222222222222222222222222222222222222

int FINUFFT2D1MANY(int n_transf, BIGINT nj, FLT *xj, FLT *yj, CPX *c, int iflag, FLT eps,
                   BIGINT ms, BIGINT mt, CPX *fk, finufft_opts *opts)
//  Type-1 2D complex nonuniform FFT, many vectors. See ../docs/usage.rst
{
  return invokeGuruInterface(2, 1, n_transf, nj, xj, yj, NULL, c, iflag, eps, {ms, mt, 1},
                             0, NULL, NULL, NULL, fk, opts);
}
int FINUFFT2D1(BIGINT nj, FLT *xj, FLT *yj, CPX *cj, int iflag, FLT eps, BIGINT ms,
               BIGINT mt, CPX *fk, finufft_opts *opts)
//  Type-1 2D complex nonuniform FFT. See ../docs/usage.rst
{
  return FINUFFT2D1MANY(1, nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts);
}

int FINUFFT2D2MANY(int n_transf, BIGINT nj, FLT *xj, FLT *yj, CPX *c, int iflag, FLT eps,
                   BIGINT ms, BIGINT mt, CPX *fk, finufft_opts *opts)
//  Type-2 2D complex nonuniform FFT, many vectors.  See ../docs/usage.rst
{
  return invokeGuruInterface(2, 2, n_transf, nj, xj, yj, NULL, c, iflag, eps, {ms, mt, 1},
                             0, NULL, NULL, NULL, fk, opts);
}
int FINUFFT2D2(BIGINT nj, FLT *xj, FLT *yj, CPX *cj, int iflag, FLT eps, BIGINT ms,
               BIGINT mt, CPX *fk, finufft_opts *opts)
//  Type-2 2D complex nonuniform FFT.  See ../docs/usage.rst
{
  return FINUFFT2D2MANY(1, nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts);
}

int FINUFFT2D3MANY(int n_transf, BIGINT nj, FLT *xj, FLT *yj, CPX *cj, int iflag, FLT eps,
                   BIGINT nk, FLT *s, FLT *t, CPX *fk, finufft_opts *opts)
// Type-3 2D complex nonuniform FFT, many vectors.  See ../docs/usage.rst
{
  return invokeGuruInterface(2, 3, n_transf, nj, xj, yj, NULL, cj, iflag, eps, {0, 0, 0},
                             nk, s, t, NULL, fk, opts);
}
int FINUFFT2D3(BIGINT nj, FLT *xj, FLT *yj, CPX *cj, int iflag, FLT eps, BIGINT nk,
               FLT *s, FLT *t, CPX *fk, finufft_opts *opts)
// Type-3 2D complex nonuniform FFT.  See ../docs/usage.rst
{
  return FINUFFT2D3MANY(1, nj, xj, yj, cj, iflag, eps, nk, s, t, fk, opts);
}

// Dimension 3333333333333333333333333333333333333333333333333333333333333333

int FINUFFT3D1MANY(int n_transf, BIGINT nj, FLT *xj, FLT *yj, FLT *zj, CPX *cj, int iflag,
                   FLT eps, BIGINT ms, BIGINT mt, BIGINT mu, CPX *fk, finufft_opts *opts)
// Type-1 3D complex nonuniform FFT, many vectors.  See ../docs/usage.rst
{
  return invokeGuruInterface(3, 1, n_transf, nj, xj, yj, zj, cj, iflag, eps, {ms, mt, mu},
                             0, NULL, NULL, NULL, fk, opts);
}
int FINUFFT3D1(BIGINT nj, FLT *xj, FLT *yj, FLT *zj, CPX *cj, int iflag, FLT eps,
               BIGINT ms, BIGINT mt, BIGINT mu, CPX *fk, finufft_opts *opts)
//  Type-1 3D complex nonuniform FFT.   See ../docs/usage.rst
{
  return FINUFFT3D1MANY(1, nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts);
}

int FINUFFT3D2MANY(int n_transf, BIGINT nj, FLT *xj, FLT *yj, FLT *zj, CPX *cj, int iflag,
                   FLT eps, BIGINT ms, BIGINT mt, BIGINT mu, CPX *fk, finufft_opts *opts)
// Type-2 3D complex nonuniform FFT, many vectors.   See ../docs/usage.rst
{
  return invokeGuruInterface(3, 2, n_transf, nj, xj, yj, zj, cj, iflag, eps, {ms, mt, mu},
                             0, NULL, NULL, NULL, fk, opts);
}
int FINUFFT3D2(BIGINT nj, FLT *xj, FLT *yj, FLT *zj, CPX *cj, int iflag, FLT eps,
               BIGINT ms, BIGINT mt, BIGINT mu, CPX *fk, finufft_opts *opts)
// Type-2 3D complex nonuniform FFT.   See ../docs/usage.rst
{
  return FINUFFT3D2MANY(1, nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts);
}

int FINUFFT3D3MANY(int n_transf, BIGINT nj, FLT *xj, FLT *yj, FLT *zj, CPX *cj, int iflag,
                   FLT eps, BIGINT nk, FLT *s, FLT *t, FLT *u, CPX *fk,
                   finufft_opts *opts)
//  Type-3 3D complex nonuniform FFT, many vectors.   See ../docs/usage.rst
{
  return invokeGuruInterface(3, 3, n_transf, nj, xj, yj, zj, cj, iflag, eps, {0, 0, 0},
                             nk, s, t, u, fk, opts);
}
int FINUFFT3D3(BIGINT nj, FLT *xj, FLT *yj, FLT *zj, CPX *cj, int iflag, FLT eps,
               BIGINT nk, FLT *s, FLT *t, FLT *u, CPX *fk, finufft_opts *opts)
//  Type-3 3D complex nonuniform FFT.   See ../docs/usage.rst
{
  return FINUFFT3D3MANY(1, nj, xj, yj, zj, cj, iflag, eps, nk, s, t, u, fk, opts);
}
