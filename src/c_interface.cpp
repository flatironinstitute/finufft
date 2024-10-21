// public header
#include <finufft.h>
// private headers
#include <array>
#include <cstdio>
#include <finufft/finufft_core.h> // (must come after complex.h)

using namespace std;

/* ---------------------------------------------------------------------------
   The 18 simple interfaces (= 3 dims * 3 types * {singlecall,many}) to FINUFFT.
   As of v1.2 these simply invoke the guru interface, through a helper layer.
   See ../docs/usage.rst or http://finufft.readthedocs.io for documentation
   all routines here.

   Authors: Andrea Malleo and Alex Barnett, 2019-2020.
   Safe namespacing, Barnett, May 2022.
   ---------------------------------------------------------------------------
*/

using f32  = float;
using f64  = double;
using c64  = std::complex<float>;
using c128 = std::complex<double>;
using i64  = BIGINT;

void finufft_default_opts(finufft_opts *o) { finufft_default_opts_t(o); }
void finufftf_default_opts(finufft_opts *o) { finufft_default_opts_t(o); }

int finufft_makeplan(int type, int dim, const i64 *n_modes, int iflag, int ntrans,
                     f64 tol, finufft_plan *pp, finufft_opts *opts) {
  return finufft_makeplan_t<f64>(type, dim, n_modes, iflag, ntrans, tol,
                                 reinterpret_cast<FINUFFT_PLAN_T<f64> **>(pp), opts);
}
int finufftf_makeplan(int type, int dim, const i64 *n_modes, int iflag, int ntrans,
                      f32 tol, finufftf_plan *pp, finufft_opts *opts) {
  return finufft_makeplan_t<f32>(type, dim, n_modes, iflag, ntrans, tol,
                                 reinterpret_cast<FINUFFT_PLAN_T<f32> **>(pp), opts);
}

int finufft_setpts(finufft_plan p, i64 nj, f64 *xj, f64 *yj, f64 *zj, i64 nk, f64 *s,
                   f64 *t, f64 *u) {
  return reinterpret_cast<FINUFFT_PLAN_T<f64> *>(p)->setpts(nj, xj, yj, zj, nk, s, t, u);
}
int finufftf_setpts(finufftf_plan p, i64 nj, f32 *xj, f32 *yj, f32 *zj, i64 nk, f32 *s,
                    f32 *t, f32 *u) {
  return reinterpret_cast<FINUFFT_PLAN_T<f32> *>(p)->setpts(nj, xj, yj, zj, nk, s, t, u);
}

int finufft_execute(finufft_plan p, c128 *cj, c128 *fk) {
  return reinterpret_cast<FINUFFT_PLAN_T<f64> *>(p)->execute(cj, fk);
}
int finufftf_execute(finufftf_plan p, c64 *cj, c64 *fk) {
  return reinterpret_cast<FINUFFT_PLAN_T<f32> *>(p)->execute(cj, fk);
}

int finufft_destroy(finufft_plan p)
// Free everything we allocated inside of finufft_plan pointed to by p.
// Also must not crash if called immediately after finufft_makeplan.
// Thus either each thing free'd here is guaranteed to be nullptr or correctly
// allocated.
{
  if (!p) // nullptr ptr, so not a ptr to a plan, report error
    return 1;

  delete reinterpret_cast<FINUFFT_PLAN_T<f64> *>(p);
  return 0; // success
}
int finufftf_destroy(finufftf_plan p)
// Free everything we allocated inside of finufft_plan pointed to by p.
// Also must not crash if called immediately after finufft_makeplan.
// Thus either each thing free'd here is guaranteed to be nullptr or correctly
// allocated.
{
  if (!p) // nullptr ptr, so not a ptr to a plan, report error
    return 1;

  delete reinterpret_cast<FINUFFT_PLAN_T<f32> *>(p);
  return 0; // success
}
// Helper layer ...........................................................

template<typename T>
static int guru(int n_dims, int type, int n_transf, i64 nj, const std::array<T *, 3> &xyz,
                std::complex<T> *cj, int iflag, T eps, const std::array<i64, 3> &n_modes,
                i64 nk, const std::array<T *, 3> &stu, std::complex<T> *fk,
                finufft_opts *popts)
// Helper layer between simple interfaces (with opts) and the guru functions.
// Author: Andrea Malleo, 2019.
{
  FINUFFT_PLAN_T<T> *plan = nullptr;
  int ier =
      finufft_makeplan_t<T>(type, n_dims, n_modes.data(), iflag, n_transf, eps, &plan,
                            popts); // popts (ptr to opts) can be nullptr
  if (ier > 1) {                    // since 1 (a warning) still allows proceeding...
    fprintf(stderr, "FINUFFT invokeGuru: plan error (ier=%d)!\n", ier);
    delete plan;
    return ier;
  }

  int ier2 = plan->setpts(nj, xyz[0], xyz[1], xyz[2], nk, stu[0], stu[1], stu[2]);
  if (ier2 > 1) {
    fprintf(stderr, "FINUFFT invokeGuru: setpts error (ier=%d)!\n", ier2);
    delete plan;
    return ier2;
  }

  int ier3 = plan->execute(cj, fk);
  if (ier3 > 1) {
    fprintf(stderr, "FINUFFT invokeGuru: execute error (ier=%d)!\n", ier3);
    delete plan;
    return ier3;
  }

  delete plan;
  return max(max(ier, ier2), ier3); // in case any one gave a (positive!) warning
}

// Dimension 1111111111111111111111111111111111111111111111111111111111111111

int finufft1d1many(int n_transf, i64 nj, f64 *xj, c128 *cj, int iflag, f64 eps, i64 ms,
                   c128 *fk, finufft_opts *opts) {
  return guru<f64>(1, 1, n_transf, nj, {xj}, cj, iflag, eps, {ms, 1, 1}, 0, {}, fk, opts);
}
int finufftf1d1many(int n_transf, i64 nj, f32 *xj, c64 *cj, int iflag, f32 eps, i64 ms,
                    c64 *fk, finufft_opts *opts) {
  return guru<f32>(1, 1, n_transf, nj, {xj}, cj, iflag, eps, {ms, 1, 1}, 0, {}, fk, opts);
}

int finufft1d1(i64 nj, f64 *xj, c128 *cj, int iflag, f64 eps, i64 ms, c128 *fk,
               finufft_opts *opts) {
  return finufft1d1many(1, nj, xj, cj, iflag, eps, ms, fk, opts);
}
int finufftf1d1(i64 nj, f32 *xj, c64 *cj, int iflag, f32 eps, i64 ms, c64 *fk,
                finufft_opts *opts) {
  return finufftf1d1many(1, nj, xj, cj, iflag, eps, ms, fk, opts);
}

int finufft1d2many(int n_transf, i64 nj, f64 *xj, c128 *cj, int iflag, f64 eps, i64 ms,
                   c128 *fk, finufft_opts *opts) {
  return guru<f64>(1, 2, n_transf, nj, {xj}, cj, iflag, eps, {ms, 1, 1}, 0, {}, fk, opts);
}
int finufftf1d2many(int n_transf, i64 nj, f32 *xj, c64 *cj, int iflag, f32 eps, i64 ms,
                    c64 *fk, finufft_opts *opts) {
  return guru<f32>(1, 2, n_transf, nj, {xj}, cj, iflag, eps, {ms, 1, 1}, 0, {}, fk, opts);
}

int finufft1d2(i64 nj, f64 *xj, c128 *cj, int iflag, f64 eps, i64 ms, c128 *fk,
               finufft_opts *opts) {
  return finufft1d2many(1, nj, xj, cj, iflag, eps, ms, fk, opts);
}
int finufftf1d2(i64 nj, f32 *xj, c64 *cj, int iflag, f32 eps, i64 ms, c64 *fk,
                finufft_opts *opts) {
  return finufftf1d2many(1, nj, xj, cj, iflag, eps, ms, fk, opts);
}

int finufft1d3many(int n_transf, i64 nj, f64 *xj, c128 *cj, int iflag, f64 eps, i64 nk,
                   f64 *s, c128 *fk, finufft_opts *opts) {
  return guru<f64>(1, 3, n_transf, nj, {xj}, cj, iflag, eps, {0, 0, 0}, nk, {s}, fk,
                   opts);
}
int finufftf1d3many(int n_transf, i64 nj, f32 *xj, c64 *cj, int iflag, f32 eps, i64 nk,
                    f32 *s, c64 *fk, finufft_opts *opts) {
  return guru<f32>(1, 3, n_transf, nj, {xj}, cj, iflag, eps, {0, 0, 0}, nk, {s}, fk,
                   opts);
}
int finufft1d3(i64 nj, f64 *xj, c128 *cj, int iflag, f64 eps, i64 nk, f64 *s, c128 *fk,
               finufft_opts *opts) {
  return finufft1d3many(1, nj, xj, cj, iflag, eps, nk, s, fk, opts);
}
int finufftf1d3(i64 nj, f32 *xj, c64 *cj, int iflag, f32 eps, i64 nk, f32 *s, c64 *fk,
                finufft_opts *opts) {
  return finufftf1d3many(1, nj, xj, cj, iflag, eps, nk, s, fk, opts);
}

// Dimension 22222222222222222222222222222222222222222222222222222222222222222

int finufft2d1many(int n_transf, i64 nj, f64 *xj, f64 *yj, c128 *c, int iflag, f64 eps,
                   i64 ms, i64 mt, c128 *fk, finufft_opts *opts) {
  return guru<f64>(2, 1, n_transf, nj, {xj, yj}, c, iflag, eps, {ms, mt, 1}, 0, {}, fk,
                   opts);
}
int finufftf2d1many(int n_transf, i64 nj, f32 *xj, f32 *yj, c64 *c, int iflag, f32 eps,
                    i64 ms, i64 mt, c64 *fk, finufft_opts *opts) {
  return guru<f32>(2, 1, n_transf, nj, {xj, yj}, c, iflag, eps, {ms, mt, 1}, 0, {}, fk,
                   opts);
}
int finufft2d1(i64 nj, f64 *xj, f64 *yj, c128 *cj, int iflag, f64 eps, i64 ms, i64 mt,
               c128 *fk, finufft_opts *opts) {
  return finufft2d1many(1, nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts);
}
int finufftf2d1(i64 nj, f32 *xj, f32 *yj, c64 *cj, int iflag, f32 eps, i64 ms, i64 mt,
                c64 *fk, finufft_opts *opts) {
  return finufftf2d1many(1, nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts);
}

int finufft2d2many(int n_transf, i64 nj, f64 *xj, f64 *yj, c128 *c, int iflag, f64 eps,
                   i64 ms, i64 mt, c128 *fk, finufft_opts *opts) {
  return guru<f64>(2, 2, n_transf, nj, {xj, yj}, c, iflag, eps, {ms, mt, 1}, 0, {}, fk,
                   opts);
}
int finufftf2d2many(int n_transf, i64 nj, f32 *xj, f32 *yj, c64 *c, int iflag, f32 eps,
                    i64 ms, i64 mt, c64 *fk, finufft_opts *opts) {
  return guru<f32>(2, 2, n_transf, nj, {xj, yj}, c, iflag, eps, {ms, mt, 1}, 0, {}, fk,
                   opts);
}
int finufft2d2(i64 nj, f64 *xj, f64 *yj, c128 *cj, int iflag, f64 eps, i64 ms, i64 mt,
               c128 *fk, finufft_opts *opts) {
  return finufft2d2many(1, nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts);
}
int finufftf2d2(i64 nj, f32 *xj, f32 *yj, c64 *cj, int iflag, f32 eps, i64 ms, i64 mt,
                c64 *fk, finufft_opts *opts) {
  return finufftf2d2many(1, nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts);
}

int finufft2d3many(int n_transf, i64 nj, f64 *xj, f64 *yj, c128 *cj, int iflag, f64 eps,
                   i64 nk, f64 *s, f64 *t, c128 *fk, finufft_opts *opts) {
  return guru<f64>(2, 3, n_transf, nj, {xj, yj}, cj, iflag, eps, {0, 0, 0}, nk, {s, t},
                   fk, opts);
}
int finufftf2d3many(int n_transf, i64 nj, f32 *xj, f32 *yj, c64 *cj, int iflag, f32 eps,
                    i64 nk, f32 *s, f32 *t, c64 *fk, finufft_opts *opts) {
  return guru<f32>(2, 3, n_transf, nj, {xj, yj}, cj, iflag, eps, {0, 0, 0}, nk, {s, t},
                   fk, opts);
}
int finufft2d3(i64 nj, f64 *xj, f64 *yj, c128 *cj, int iflag, f64 eps, i64 nk, f64 *s,
               f64 *t, c128 *fk, finufft_opts *opts) {
  return finufft2d3many(1, nj, xj, yj, cj, iflag, eps, nk, s, t, fk, opts);
}
int finufftf2d3(i64 nj, f32 *xj, f32 *yj, c64 *cj, int iflag, f32 eps, i64 nk, f32 *s,
                f32 *t, c64 *fk, finufft_opts *opts) {
  return finufftf2d3many(1, nj, xj, yj, cj, iflag, eps, nk, s, t, fk, opts);
}

// Dimension 3333333333333333333333333333333333333333333333333333333333333333

int finufft3d1many(int n_transf, i64 nj, f64 *xj, f64 *yj, f64 *zj, c128 *cj, int iflag,
                   f64 eps, i64 ms, i64 mt, i64 mu, c128 *fk, finufft_opts *opts) {
  return guru<f64>(3, 1, n_transf, nj, {xj, yj, zj}, cj, iflag, eps, {ms, mt, mu}, 0, {},
                   fk, opts);
}
int finufftf3d1many(int n_transf, i64 nj, f32 *xj, f32 *yj, f32 *zj, c64 *cj, int iflag,
                    f32 eps, i64 ms, i64 mt, i64 mu, c64 *fk, finufft_opts *opts) {
  return guru<f32>(3, 1, n_transf, nj, {xj, yj, zj}, cj, iflag, eps, {ms, mt, mu}, 0, {},
                   fk, opts);
}
int finufft3d1(i64 nj, f64 *xj, f64 *yj, f64 *zj, c128 *cj, int iflag, f64 eps, i64 ms,
               i64 mt, i64 mu, c128 *fk, finufft_opts *opts) {
  return finufft3d1many(1, nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts);
}
int finufftf3d1(i64 nj, f32 *xj, f32 *yj, f32 *zj, c64 *cj, int iflag, f32 eps, i64 ms,
                i64 mt, i64 mu, c64 *fk, finufft_opts *opts) {
  return finufftf3d1many(1, nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts);
}

int finufft3d2many(int n_transf, i64 nj, f64 *xj, f64 *yj, f64 *zj, c128 *cj, int iflag,
                   f64 eps, i64 ms, i64 mt, i64 mu, c128 *fk, finufft_opts *opts) {
  return guru<f64>(3, 2, n_transf, nj, {xj, yj, zj}, cj, iflag, eps, {ms, mt, mu}, 0, {},
                   fk, opts);
}
int finufftf3d2many(int n_transf, i64 nj, f32 *xj, f32 *yj, f32 *zj, c64 *cj, int iflag,
                    f32 eps, i64 ms, i64 mt, i64 mu, c64 *fk, finufft_opts *opts) {
  return guru<f32>(3, 2, n_transf, nj, {xj, yj, zj}, cj, iflag, eps, {ms, mt, mu}, 0, {},
                   fk, opts);
}
int finufft3d2(i64 nj, f64 *xj, f64 *yj, f64 *zj, c128 *cj, int iflag, f64 eps, i64 ms,
               i64 mt, i64 mu, c128 *fk, finufft_opts *opts) {
  return finufft3d2many(1, nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts);
}
int finufftf3d2(i64 nj, f32 *xj, f32 *yj, f32 *zj, c64 *cj, int iflag, f32 eps, i64 ms,
                i64 mt, i64 mu, c64 *fk, finufft_opts *opts) {
  return finufftf3d2many(1, nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts);
}

int finufft3d3many(int n_transf, i64 nj, f64 *xj, f64 *yj, f64 *zj, c128 *cj, int iflag,
                   f64 eps, i64 nk, f64 *s, f64 *t, f64 *u, c128 *fk,
                   finufft_opts *opts) {
  return guru<f64>(3, 3, n_transf, nj, {xj, yj, zj}, cj, iflag, eps, {0, 0, 0}, nk,
                   {s, t, u}, fk, opts);
}
int finufftf3d3many(int n_transf, i64 nj, f32 *xj, f32 *yj, f32 *zj, c64 *cj, int iflag,
                    f32 eps, i64 nk, f32 *s, f32 *t, f32 *u, c64 *fk,
                    finufft_opts *opts) {
  return guru<f32>(3, 3, n_transf, nj, {xj, yj, zj}, cj, iflag, eps, {0, 0, 0}, nk,
                   {s, t, u}, fk, opts);
}
int finufft3d3(i64 nj, f64 *xj, f64 *yj, f64 *zj, c128 *cj, int iflag, f64 eps, i64 nk,
               f64 *s, f64 *t, f64 *u, c128 *fk, finufft_opts *opts) {
  return finufft3d3many(1, nj, xj, yj, zj, cj, iflag, eps, nk, s, t, u, fk, opts);
}
int finufftf3d3(i64 nj, f32 *xj, f32 *yj, f32 *zj, c64 *cj, int iflag, f32 eps, i64 nk,
                f32 *s, f32 *t, f32 *u, c64 *fk, finufft_opts *opts) {
  return finufftf3d3many(1, nj, xj, yj, zj, cj, iflag, eps, nk, s, t, u, fk, opts);
}
