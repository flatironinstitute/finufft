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
   This compiles in either double or single precision (based on -DSINGLE),
   producing functions finufft?d?{many} or finufftf?1?{many} respectively.

   Authors: Andrea Malleo and Alex Barnett, 2019-2020.
   Safe namespacing, Barnett, May 2022.
   ---------------------------------------------------------------------------
*/

void finufft_default_opts(finufft_opts *o) { finufft_default_opts_t(o); }
void finufftf_default_opts(finufft_opts *o) { finufft_default_opts_t(o); }

int finufft_makeplan(int type, int dim, const BIGINT *n_modes, int iflag, int ntrans,
                     double tol, finufft_plan *pp, finufft_opts *opts) {
  return finufft_makeplan_t<double>(type, dim, n_modes, iflag, ntrans, tol,
                                    reinterpret_cast<FINUFFT_PLAN_T<double> **>(pp),
                                    opts);
}
int finufftf_makeplan(int type, int dim, const BIGINT *n_modes, int iflag, int ntrans,
                      float tol, finufftf_plan *pp, finufft_opts *opts) {
  return finufft_makeplan_t<float>(type, dim, n_modes, iflag, ntrans, tol,
                                   reinterpret_cast<FINUFFT_PLAN_T<float> **>(pp), opts);
}

int finufft_setpts(finufft_plan p, BIGINT nj, double *xj, double *yj, double *zj,
                   BIGINT nk, double *s, double *t, double *u) {
  return finufft_setpts_t<double>(reinterpret_cast<FINUFFT_PLAN_T<double> *>(p), nj, xj,
                                  yj, zj, nk, s, t, u);
}
int finufftf_setpts(finufftf_plan p, BIGINT nj, float *xj, float *yj, float *zj,
                    BIGINT nk, float *s, float *t, float *u) {
  return finufft_setpts_t<float>(reinterpret_cast<FINUFFT_PLAN_T<float> *>(p), nj, xj, yj,
                                 zj, nk, s, t, u);
}

int finufft_execute(finufft_plan p, std::complex<double> *cj, std::complex<double> *fk) {
  return finufft_execute_t<double>(reinterpret_cast<FINUFFT_PLAN_T<double> *>(p), cj, fk);
}
int finufftf_execute(finufftf_plan p, std::complex<float> *cj, std::complex<float> *fk) {
  return finufft_execute_t<float>(reinterpret_cast<FINUFFT_PLAN_T<float> *>(p), cj, fk);
}

int finufft_destroy(finufft_plan p)
// Free everything we allocated inside of finufft_plan pointed to by p.
// Also must not crash if called immediately after finufft_makeplan.
// Thus either each thing free'd here is guaranteed to be nullptr or correctly
// allocated.
{
  if (!p) // nullptr ptr, so not a ptr to a plan, report error
    return 1;

  delete reinterpret_cast<FINUFFT_PLAN_T<double> *>(p);
  p = nullptr;
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

  delete reinterpret_cast<FINUFFT_PLAN_T<float> *>(p);
  p = nullptr;
  return 0; // success
}
// Helper layer ...........................................................

namespace finufft {
namespace common {

template<typename T>
static int invokeGuruInterface(int n_dims, int type, int n_transf, BIGINT nj, T *xj,
                               T *yj, T *zj, std::complex<T> *cj, int iflag, T eps,
                               const std::array<BIGINT, 3> &n_modes, BIGINT nk, T *s,
                               T *t, T *u, std::complex<T> *fk, finufft_opts *popts)
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

  int ier2 = finufft_setpts_t<T>(plan, nj, xj, yj, zj, nk, s, t, u);
  if (ier2 > 1) {
    fprintf(stderr, "FINUFFT invokeGuru: setpts error (ier=%d)!\n", ier2);
    delete plan;
    return ier2;
  }

  int ier3 = finufft_execute_t<T>(plan, cj, fk);
  if (ier3 > 1) {
    fprintf(stderr, "FINUFFT invokeGuru: execute error (ier=%d)!\n", ier3);
    delete plan;
    return ier3;
  }

  delete plan;
  return max(max(ier, ier2), ier3); // in case any one gave a (positive!) warning
}

} // namespace common
} // namespace finufft

using namespace finufft::common;

// Dimension 1111111111111111111111111111111111111111111111111111111111111111

int finufft1d1many(int n_transf, BIGINT nj, double *xj, std::complex<double> *cj,
                   int iflag, double eps, BIGINT ms, std::complex<double> *fk,
                   finufft_opts *opts)
// Type-1 1D complex nonuniform FFT for many vectors. See ../docs/usage.rst
{
  return invokeGuruInterface<double>(1, 1, n_transf, nj, xj, nullptr, nullptr, cj, iflag,
                                     eps, {ms, 1, 1}, 0, nullptr, nullptr, nullptr, fk,
                                     opts);
}
int finufftf1d1many(int n_transf, BIGINT nj, float *xj, std::complex<float> *cj,
                    int iflag, float eps, BIGINT ms, std::complex<float> *fk,
                    finufft_opts *opts)
// Type-1 1D complex nonuniform FFT for many vectors. See ../docs/usage.rst
{
  return invokeGuruInterface<float>(1, 1, n_transf, nj, xj, nullptr, nullptr, cj, iflag,
                                    eps, {ms, 1, 1}, 0, nullptr, nullptr, nullptr, fk,
                                    opts);
}

int finufft1d1(BIGINT nj, double *xj, std::complex<double> *cj, int iflag, double eps,
               BIGINT ms, std::complex<double> *fk, finufft_opts *opts)
//  Type-1 1D complex nonuniform FFT. See ../docs/usage.rst
{
  return finufft1d1many(1, nj, xj, cj, iflag, eps, ms, fk, opts);
}
int finufftf1d1(BIGINT nj, float *xj, std::complex<float> *cj, int iflag, float eps,
                BIGINT ms, std::complex<float> *fk, finufft_opts *opts)
//  Type-1 1D complex nonuniform FFT. See ../docs/usage.rst
{
  return finufftf1d1many(1, nj, xj, cj, iflag, eps, ms, fk, opts);
}

int finufft1d2many(int n_transf, BIGINT nj, double *xj, std::complex<double> *cj,
                   int iflag, double eps, BIGINT ms, std::complex<double> *fk,
                   finufft_opts *opts)
//  Type-2 1D complex nonuniform FFT, many vectors. See ../docs/usage.rst
{
  return invokeGuruInterface<double>(1, 2, n_transf, nj, xj, nullptr, nullptr, cj, iflag,
                                     eps, {ms, 1, 1}, 0, nullptr, nullptr, nullptr, fk,
                                     opts);
}
int finufftf1d2many(int n_transf, BIGINT nj, float *xj, std::complex<float> *cj,
                    int iflag, float eps, BIGINT ms, std::complex<float> *fk,
                    finufft_opts *opts)
//  Type-2 1D complex nonuniform FFT, many vectors. See ../docs/usage.rst
{
  return invokeGuruInterface<float>(1, 2, n_transf, nj, xj, nullptr, nullptr, cj, iflag,
                                    eps, {ms, 1, 1}, 0, nullptr, nullptr, nullptr, fk,
                                    opts);
}

int finufft1d2(BIGINT nj, double *xj, std::complex<double> *cj, int iflag, double eps,
               BIGINT ms, std::complex<double> *fk, finufft_opts *opts)
//  Type-2 1D complex nonuniform FFT. See ../docs/usage.rst
{
  return finufft1d2many(1, nj, xj, cj, iflag, eps, ms, fk, opts);
}
int finufftf1d2(BIGINT nj, float *xj, std::complex<float> *cj, int iflag, float eps,
                BIGINT ms, std::complex<float> *fk, finufft_opts *opts)
//  Type-2 1D complex nonuniform FFT. See ../docs/usage.rst
{
  return finufftf1d2many(1, nj, xj, cj, iflag, eps, ms, fk, opts);
}

int finufft1d3many(int n_transf, BIGINT nj, double *xj, std::complex<double> *cj,
                   int iflag, double eps, BIGINT nk, double *s, std::complex<double> *fk,
                   finufft_opts *opts)
// Type-3 1D complex nonuniform FFT, many vectors. See ../docs/usage.rst
{
  return invokeGuruInterface<double>(1, 3, n_transf, nj, xj, nullptr, nullptr, cj, iflag,
                                     eps, {0, 0, 0}, nk, s, nullptr, nullptr, fk, opts);
}
int finufftf1d3many(int n_transf, BIGINT nj, float *xj, std::complex<float> *cj,
                    int iflag, float eps, BIGINT nk, float *s, std::complex<float> *fk,
                    finufft_opts *opts)
// Type-3 1D complex nonuniform FFT, many vectors. See ../docs/usage.rst
{
  return invokeGuruInterface<float>(1, 3, n_transf, nj, xj, nullptr, nullptr, cj, iflag,
                                    eps, {0, 0, 0}, nk, s, nullptr, nullptr, fk, opts);
}
int finufft1d3(BIGINT nj, double *xj, std::complex<double> *cj, int iflag, double eps,
               BIGINT nk, double *s, std::complex<double> *fk, finufft_opts *opts)
// Type-3 1D complex nonuniform FFT. See ../docs/usage.rst
{
  return finufft1d3many(1, nj, xj, cj, iflag, eps, nk, s, fk, opts);
}
int finufftf1d3(BIGINT nj, float *xj, std::complex<float> *cj, int iflag, float eps,
                BIGINT nk, float *s, std::complex<float> *fk, finufft_opts *opts)
// Type-3 1D complex nonuniform FFT. See ../docs/usage.rst
{
  return finufftf1d3many(1, nj, xj, cj, iflag, eps, nk, s, fk, opts);
}

// Dimension 22222222222222222222222222222222222222222222222222222222222222222

int finufft2d1many(int n_transf, BIGINT nj, double *xj, double *yj,
                   std::complex<double> *c, int iflag, double eps, BIGINT ms, BIGINT mt,
                   std::complex<double> *fk, finufft_opts *opts)
//  Type-1 2D complex nonuniform FFT, many vectors. See ../docs/usage.rst
{
  return invokeGuruInterface<double>(2, 1, n_transf, nj, xj, yj, nullptr, c, iflag, eps,
                                     {ms, mt, 1}, 0, nullptr, nullptr, nullptr, fk, opts);
}
int finufftf2d1many(int n_transf, BIGINT nj, float *xj, float *yj, std::complex<float> *c,
                    int iflag, float eps, BIGINT ms, BIGINT mt, std::complex<float> *fk,
                    finufft_opts *opts)
//  Type-1 2D complex nonuniform FFT, many vectors. See ../docs/usage.rst
{
  return invokeGuruInterface<float>(2, 1, n_transf, nj, xj, yj, nullptr, c, iflag, eps,
                                    {ms, mt, 1}, 0, nullptr, nullptr, nullptr, fk, opts);
}
int finufft2d1(BIGINT nj, double *xj, double *yj, std::complex<double> *cj, int iflag,
               double eps, BIGINT ms, BIGINT mt, std::complex<double> *fk,
               finufft_opts *opts)
//  Type-1 2D complex nonuniform FFT. See ../docs/usage.rst
{
  return finufft2d1many(1, nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts);
}
int finufftf2d1(BIGINT nj, float *xj, float *yj, std::complex<float> *cj, int iflag,
                float eps, BIGINT ms, BIGINT mt, std::complex<float> *fk,
                finufft_opts *opts)
//  Type-1 2D complex nonuniform FFT. See ../docs/usage.rst
{
  return finufftf2d1many(1, nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts);
}

int finufft2d2many(int n_transf, BIGINT nj, double *xj, double *yj,
                   std::complex<double> *c, int iflag, double eps, BIGINT ms, BIGINT mt,
                   std::complex<double> *fk, finufft_opts *opts)
//  Type-2 2D complex nonuniform FFT, many vectors.  See ../docs/usage.rst
{
  return invokeGuruInterface<double>(2, 2, n_transf, nj, xj, yj, nullptr, c, iflag, eps,
                                     {ms, mt, 1}, 0, nullptr, nullptr, nullptr, fk, opts);
}
int finufftf2d2many(int n_transf, BIGINT nj, float *xj, float *yj, std::complex<float> *c,
                    int iflag, float eps, BIGINT ms, BIGINT mt, std::complex<float> *fk,
                    finufft_opts *opts)
//  Type-2 2D complex nonuniform FFT, many vectors.  See ../docs/usage.rst
{
  return invokeGuruInterface<float>(2, 2, n_transf, nj, xj, yj, nullptr, c, iflag, eps,
                                    {ms, mt, 1}, 0, nullptr, nullptr, nullptr, fk, opts);
}
int finufft2d2(BIGINT nj, double *xj, double *yj, std::complex<double> *cj, int iflag,
               double eps, BIGINT ms, BIGINT mt, std::complex<double> *fk,
               finufft_opts *opts)
//  Type-2 2D complex nonuniform FFT.  See ../docs/usage.rst
{
  return finufft2d2many(1, nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts);
}
int finufftf2d2(BIGINT nj, float *xj, float *yj, std::complex<float> *cj, int iflag,
                float eps, BIGINT ms, BIGINT mt, std::complex<float> *fk,
                finufft_opts *opts)
//  Type-2 2D complex nonuniform FFT.  See ../docs/usage.rst
{
  return finufftf2d2many(1, nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts);
}

int finufft2d3many(int n_transf, BIGINT nj, double *xj, double *yj,
                   std::complex<double> *cj, int iflag, double eps, BIGINT nk, double *s,
                   double *t, std::complex<double> *fk, finufft_opts *opts)
// Type-3 2D complex nonuniform FFT, many vectors.  See ../docs/usage.rst
{
  return invokeGuruInterface<double>(2, 3, n_transf, nj, xj, yj, nullptr, cj, iflag, eps,
                                     {0, 0, 0}, nk, s, t, nullptr, fk, opts);
}
int finufftf2d3many(int n_transf, BIGINT nj, float *xj, float *yj,
                    std::complex<float> *cj, int iflag, float eps, BIGINT nk, float *s,
                    float *t, std::complex<float> *fk, finufft_opts *opts)
// Type-3 2D complex nonuniform FFT, many vectors.  See ../docs/usage.rst
{
  return invokeGuruInterface<float>(2, 3, n_transf, nj, xj, yj, nullptr, cj, iflag, eps,
                                    {0, 0, 0}, nk, s, t, nullptr, fk, opts);
}
int finufft2d3(BIGINT nj, double *xj, double *yj, std::complex<double> *cj, int iflag,
               double eps, BIGINT nk, double *s, double *t, std::complex<double> *fk,
               finufft_opts *opts)
// Type-3 2D complex nonuniform FFT.  See ../docs/usage.rst
{
  return finufft2d3many(1, nj, xj, yj, cj, iflag, eps, nk, s, t, fk, opts);
}
int finufftf2d3(BIGINT nj, float *xj, float *yj, std::complex<float> *cj, int iflag,
                float eps, BIGINT nk, float *s, float *t, std::complex<float> *fk,
                finufft_opts *opts)
// Type-3 2D complex nonuniform FFT.  See ../docs/usage.rst
{
  return finufftf2d3many(1, nj, xj, yj, cj, iflag, eps, nk, s, t, fk, opts);
}

// Dimension 3333333333333333333333333333333333333333333333333333333333333333

int finufft3d1many(int n_transf, BIGINT nj, double *xj, double *yj, double *zj,
                   std::complex<double> *cj, int iflag, double eps, BIGINT ms, BIGINT mt,
                   BIGINT mu, std::complex<double> *fk, finufft_opts *opts)
// Type-1 3D complex nonuniform FFT, many vectors.  See ../docs/usage.rst
{
  return invokeGuruInterface<double>(3, 1, n_transf, nj, xj, yj, zj, cj, iflag, eps,
                                     {ms, mt, mu}, 0, nullptr, nullptr, nullptr, fk,
                                     opts);
}
int finufftf3d1many(int n_transf, BIGINT nj, float *xj, float *yj, float *zj,
                    std::complex<float> *cj, int iflag, float eps, BIGINT ms, BIGINT mt,
                    BIGINT mu, std::complex<float> *fk, finufft_opts *opts)
// Type-1 3D complex nonuniform FFT, many vectors.  See ../docs/usage.rst
{
  return invokeGuruInterface<float>(3, 1, n_transf, nj, xj, yj, zj, cj, iflag, eps,
                                    {ms, mt, mu}, 0, nullptr, nullptr, nullptr, fk, opts);
}
int finufft3d1(BIGINT nj, double *xj, double *yj, double *zj, std::complex<double> *cj,
               int iflag, double eps, BIGINT ms, BIGINT mt, BIGINT mu,
               std::complex<double> *fk, finufft_opts *opts)
//  Type-1 3D complex nonuniform FFT.   See ../docs/usage.rst
{
  return finufft3d1many(1, nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts);
}
int finufftf3d1(BIGINT nj, float *xj, float *yj, float *zj, std::complex<float> *cj,
                int iflag, float eps, BIGINT ms, BIGINT mt, BIGINT mu,
                std::complex<float> *fk, finufft_opts *opts)
//  Type-1 3D complex nonuniform FFT.   See ../docs/usage.rst
{
  return finufftf3d1many(1, nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts);
}

int finufft3d2many(int n_transf, BIGINT nj, double *xj, double *yj, double *zj,
                   std::complex<double> *cj, int iflag, double eps, BIGINT ms, BIGINT mt,
                   BIGINT mu, std::complex<double> *fk, finufft_opts *opts)
// Type-2 3D complex nonuniform FFT, many vectors.   See ../docs/usage.rst
{
  return invokeGuruInterface<double>(3, 2, n_transf, nj, xj, yj, zj, cj, iflag, eps,
                                     {ms, mt, mu}, 0, nullptr, nullptr, nullptr, fk,
                                     opts);
}
int finufftf3d2many(int n_transf, BIGINT nj, float *xj, float *yj, float *zj,
                    std::complex<float> *cj, int iflag, float eps, BIGINT ms, BIGINT mt,
                    BIGINT mu, std::complex<float> *fk, finufft_opts *opts)
// Type-2 3D complex nonuniform FFT, many vectors.   See ../docs/usage.rst
{
  return invokeGuruInterface<float>(3, 2, n_transf, nj, xj, yj, zj, cj, iflag, eps,
                                    {ms, mt, mu}, 0, nullptr, nullptr, nullptr, fk, opts);
}
int finufft3d2(BIGINT nj, double *xj, double *yj, double *zj, std::complex<double> *cj,
               int iflag, double eps, BIGINT ms, BIGINT mt, BIGINT mu,
               std::complex<double> *fk, finufft_opts *opts)
// Type-2 3D complex nonuniform FFT.   See ../docs/usage.rst
{
  return finufft3d2many(1, nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts);
}
int finufftf3d2(BIGINT nj, float *xj, float *yj, float *zj, std::complex<float> *cj,
                int iflag, float eps, BIGINT ms, BIGINT mt, BIGINT mu,
                std::complex<float> *fk, finufft_opts *opts)
// Type-2 3D complex nonuniform FFT.   See ../docs/usage.rst
{
  return finufftf3d2many(1, nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts);
}

int finufft3d3many(int n_transf, BIGINT nj, double *xj, double *yj, double *zj,
                   std::complex<double> *cj, int iflag, double eps, BIGINT nk, double *s,
                   double *t, double *u, std::complex<double> *fk, finufft_opts *opts)
//  Type-3 3D complex nonuniform FFT, many vectors.   See ../docs/usage.rst
{
  return invokeGuruInterface<double>(3, 3, n_transf, nj, xj, yj, zj, cj, iflag, eps,
                                     {0, 0, 0}, nk, s, t, u, fk, opts);
}
int finufftf3d3many(int n_transf, BIGINT nj, float *xj, float *yj, float *zj,
                    std::complex<float> *cj, int iflag, float eps, BIGINT nk, float *s,
                    float *t, float *u, std::complex<float> *fk, finufft_opts *opts)
//  Type-3 3D complex nonuniform FFT, many vectors.   See ../docs/usage.rst
{
  return invokeGuruInterface<float>(3, 3, n_transf, nj, xj, yj, zj, cj, iflag, eps,
                                    {0, 0, 0}, nk, s, t, u, fk, opts);
}
int finufft3d3(BIGINT nj, double *xj, double *yj, double *zj, std::complex<double> *cj,
               int iflag, double eps, BIGINT nk, double *s, double *t, double *u,
               std::complex<double> *fk, finufft_opts *opts)
//  Type-3 3D complex nonuniform FFT.   See ../docs/usage.rst
{
  return finufft3d3many(1, nj, xj, yj, zj, cj, iflag, eps, nk, s, t, u, fk, opts);
}
int finufftf3d3(BIGINT nj, float *xj, float *yj, float *zj, std::complex<float> *cj,
                int iflag, float eps, BIGINT nk, float *s, float *t, float *u,
                std::complex<float> *fk, finufft_opts *opts)
//  Type-3 3D complex nonuniform FFT.   See ../docs/usage.rst
{
  return finufftf3d3many(1, nj, xj, yj, zj, cj, iflag, eps, nk, s, t, u, fk, opts);
}
