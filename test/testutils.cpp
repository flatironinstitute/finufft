/* unit tests for utils module.

   Usage: ./testutils{f}

   Pass: exit code 0. (Stdout should indicate passed)
   Fail: exit code>0. (Stdout may indicate what failed)

   June 2023: switched to pass-fail tests within the executable (more clear,
   and platform-indep, than having to compare the text output)

   Suggested compile. double-prec:
   g++ -std=c++17 -fopenmp testutils.cpp -I../include ../src/utils.o
       ../src/utils.o -o testutils -lgomp
   single-prec:
   g++ -std=c++17 -fopenmp testutils.cpp
       -I../include ../src/utils.o -o testutilsf -lgomp -DSINGLE
*/

// This switches FLT macro from double to float if SINGLE is defined, etc...

#include "finufft/utils.hpp"
#include "utils/norms.hpp"
#include <finufft/heuristics.hpp> // complexity-based upsampfac (sigma) picker
#include <finufft/test_defs.hpp>

using namespace finufft::common;
using namespace finufft::heuristics;

int main() {
#ifdef SINGLE
  printf("testutilsf started...\n");
#else
  printf("testutils started...\n");
#endif

  // test next235...
  // Barnett 2/9/17, made smaller range 3/28/17. pass-fail 6/16/23
  // The true outputs from {0,1,..,99}:
  const BIGINT next235even_true[100] = {
      2,  2,  2,  4,  4,  6,  6,  8,  8,  10, 10, 12, 12, 16, 16, 16, 16, 18,  18,  20,
      20, 24, 24, 24, 24, 30, 30, 30, 30, 30, 30, 32, 32, 36, 36, 36, 36, 40,  40,  40,
      40, 48, 48, 48, 48, 48, 48, 48, 48, 50, 50, 54, 54, 54, 54, 60, 60, 60,  60,  60,
      60, 64, 64, 64, 64, 72, 72, 72, 72, 72, 72, 72, 72, 80, 80, 80, 80, 80,  80,  80,
      80, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 96, 96, 96, 96, 96, 96, 100, 100, 100};
  for (BIGINT n = 0; n < 100; ++n) {
    BIGINT o = next235(n, 2);
    BIGINT t = next235even_true[n];
    if (o != t) {
      printf("next235(%lld, 2) =\t%lld, error should be %lld!\n", (long long)n,
             (long long)o, (long long)t);
      return 1;
    }
  }
  // a grid dimension above 2^31, which MAX_NF (1e12) allows. Catches a next235
  // narrower than BIGINT, which on Windows a plain long is.
  {
    const BIGINT n = BIGINT(3e9), o = next235(n, 2);
    if (o < n) {
      printf("next235(%lld, 2) =\t%lld, error should be >= the input!\n", (long long)n,
             (long long)o);
      return 1;
    }
  }
  // various old devel expts and comments for next235even...
  // printf("starting huge next235even...\n");   // 1e11 takes 1 sec
  // BIGINT n=(BIGINT)120573851963;
  // printf("next235even(%ld) =\t%ld\n",n,next235even(n));
  // double* a; printf("%g\n",a[0]);  // do deliberate segfault for bash debug!

  // test Gauss-Legendre quadrature...
  const int n = 16;
  std::vector<double> x(n), w(n);
  finufft::common::gaussquad(n, x.data(), w.data());
  auto f = [](double x) {
    return sin(4 * x + 1.0) + 0.3;
  }; // a test func f(x)
  auto fp = [](double x) {
    return 4 * cos(4 * x + 1.0);
  }; // its deriv f'(x)
  double I = 0;
  for (int i = 0; i < n; ++i) I += w[i] * fp(x[i]);
  double Iex = f(1.0) - f(-1.0);
  double err = std::abs(I - Iex);
  if (err > 1e-14) { // for the above func, err should be 4e-14
    printf("fail: gaussquad error %g\n", err);
    return 1;
  }

  // test vector norms and norm difference routines... now pass-fail 6/16/23
  BIGINT M = 1e4;
  std::vector<CPX> a(M), b(M);
  for (BIGINT j = 0; j < M; ++j) {
    a[j] = CPX(1.0, 0.0);
    b[j] = a[j];
  }
  constexpr FLT EPSILON = std::numeric_limits<FLT>::epsilon();
  FLT relerr            = 2.0 * EPSILON; // 1 ULP, fine since 1.0 rep exactly
  if (std::abs(infnorm(M, &a[0]) - 1.0) > relerr) return 1;
  if (std::abs(twonorm(M, &a[0]) - std::sqrt((FLT)M)) > relerr * std::sqrt((FLT)M)) return 1;
  b[0] = CPX(0.0, 0.0); // perturb b from a
  if (std::abs(errtwonorm(M, &a[0], &b[0]) - 1.0) > relerr) return 1;
  if (std::abs(std::sqrt((FLT)M) * relerrtwonorm(M, &a[0], &b[0]) - 1.0) > relerr) return 1;

#ifndef SINGLE
  // Complexity-based upsampfac (sigma) picker (finufft/heuristics.hpp). The block
  // exercises both precisions explicitly, so it runs once in the double build.
  {
    const double eps_d = std::numeric_limits<double>::epsilon();
    const double eps_f = std::numeric_limits<float>::epsilon();
    const int ns_f      = MAX_NSPREAD<float>;

    // (A) ns is non-increasing as sigma rises (the minimizer enumerates one candidate
    // per achievable width). Double holds over the whole auto range. Float holds only
    // where the cancellation guard in clamp_kernel_ns (kernel.h) does not cap ns: the
    // cap rises with sigma, so a capped ns can step up.
    const double tols[] = {1e-3, 1e-6, 1e-10, 1e-13};
    for (int dim = 1; dim <= 3; ++dim)
      for (int type = 1; type <= 3; ++type)
        for (double tol : tols) {
          int prev_d = 1 << 30, prev_f = 1 << 30;
          for (double s = MIN_AUTO_UPSAMPFAC; s <= MAX_AUTO_UPSAMPFAC + 1e-9; s += 0.05) {
            const int nd = kernel_width_at<double>(tol, dim, type, s);
            if (nd > prev_d) {
              printf("fail: ns(double) rose: dim=%d type=%d tol=%.0e sigma=%.2f\n", dim,
                     type, tol, s);
              return 1;
            }
            prev_d       = nd;
            const int nf = kernel_width_at<float>(tol, dim, type, s);
            // skip float sigmas where the cancellation guard caps ns (low-sigma prefix)
            if (nf <
                std::min(finufft::kernel::theoretical_kernel_ns(tol, dim, type, s), ns_f))
              continue;
            if (nf > prev_f) {
              printf("fail: ns(float) rose: dim=%d type=%d tol=%.0e sigma=%.2f\n", dim,
                     type, tol, s);
              return 1;
            }
            prev_f = nf;
          }
        }

    // (B) The narrow-kernel lever is real: at tight tol, ns strictly drops from
    // sigma 2.0 to 2.5 (double, dim 3), so higher sigma can pay off.
    if (!(kernel_width_at<double>(1e-13, 3, 1, 2.5) <
          kernel_width_at<double>(1e-13, 3, 1, 2.0))) {
      printf("fail: expected ns(2.5) < ns(2.0) at tol=1e-13 dim=3\n");
      return 1;
    }

    // (C) For eps_round <= tol <= 1e3 eps_round, analytic_upsampfac returns a feasible
    // sigma in [MIN_AUTO_UPSAMPFAC, MAX_AUTO_UPSAMPFAC] for every dim/type, both
    // precisions, short and long grids. A fixed sigma = 2.5 is not enough here: E makes
    // it infeasible on short grids ((F) below).
    for (int dim = 1; dim <= 3; ++dim)
      for (int type = 1; type <= 3; ++type)
        for (double maxN : {16.0, 256.0, 1e5})
          for (int is_float = 0; is_float <= 1; ++is_float)
            for (int k = 0; k < 40; ++k) {
              const double eps_round =
                  rounding_floor(is_float ? eps_f : eps_d, maxN, dim);
              const double tol = eps_round * std::pow(1e3, k / 39.0);
              const double s   = analytic_upsampfac(tol, dim, type, is_float, maxN);
              if (s < MIN_AUTO_UPSAMPFAC || s > MAX_AUTO_UPSAMPFAC ||
                  !upsampfac_feasible(s, tol, dim, type, is_float, maxN)) {
                printf("fail: analytic sigma %.3f infeasible: dim=%d type=%d N=%g %s "
                       "tol=%.3g\n",
                       s, dim, type, maxN, is_float ? "float" : "double", tol);
                return 1;
              }
            }

    // (G) minimize scores type 3 at each width edge smallest_sigma_for_ns(ns_t); every
    // edge must be feasible. Control: without the 1e-12 step, some edges must fail.
    {
      int unstepped_fails = 0;
      for (int dim = 1; dim <= 3; ++dim)
        for (int k = 3; k <= 13; ++k)
          for (int ns_t = 2; ns_t <= MAX_NSPREAD<double>; ++ns_t) {
            const double tol = std::pow(10.0, -k);
            const double s   = smallest_sigma_for_ns(tol, dim, 3, ns_t);
            if (s < MIN_AUTO_UPSAMPFAC || s >= MAX_AUTO_UPSAMPFAC) continue;
            if (!upsampfac_feasible(s, tol, dim, 3, false, 1.0)) {
              printf("fail: type 3 sigma %.6f (ns %d) infeasible: dim=%d tol=%.3g\n", s,
                     ns_t, dim, tol);
              return 1;
            }
            unstepped_fails +=
                !upsampfac_feasible(s / (1 + 1e-12), tol, dim, 3, false, 1.0);
          }
      printf("(G) %d unstepped edge sigmas infeasible, all stepped ones feasible\n",
             unstepped_fails);
      if (unstepped_fails == 0) {
        printf("fail: (G) control: no unstepped edge sigma is infeasible\n");
        return 1;
      }
    }

    // (F) eps_alias/tol = 42, 0.18, 1.05 at sigma = 1.25, 2.0, 2.5: want
    // eps_alias(analytic_upsampfac) <= tol.
    {
      const double maxN   = 16;
      const double tol    = 5.6234e-14;
      const auto feasible = [&](double s) {
        return upsampfac_feasible(s, tol, 2, 1, false, maxN);
      };
      if (feasible(MIN_AUTO_UPSAMPFAC) || feasible(MAX_AUTO_UPSAMPFAC) ||
          !feasible(2.0)) {
        printf("fail: (F) cell no longer has an interior-only feasible sigma\n");
        return 1;
      }
      const double s = analytic_upsampfac(tol, 2, 1, false, maxN);
      if (!feasible(s)) {
        printf("fail: (F) analytic sigma %.3f not feasible\n", s);
        return 1;
      }
      printf("(F) interior-only feasible cell: analytic sigma %.3f\n", s);
    }

    // (E) Density drives the pick: a spread-dominated transform (many points, small
    // grid) chooses a larger sigma than an FFT-dominated one (few points, large grid).
    {
      const int dim = 3, type = 1, nthr = 1;
      const double tol = 1e-13; // tight enough that ns drops across [2.0,2.5]
      const double dense_modes[3] = {64, 64, 64};
      const double sparse_modes[3] = {512, 512, 512};
      const double sigma_dense =
          best_type12<double>(tol, dim, type, nthr, dense_modes, /*npts=*/5e7).sigma;
      const double sigma_sparse =
          best_type12<double>(tol, dim, type, nthr, sparse_modes, /*npts=*/1e3).sigma;
      if (!(sigma_dense > sigma_sparse) || !(sigma_dense > 2.0 - 1e-9)) {
        printf("fail: dense sigma (%.3f) should exceed sparse (%.3f) and 2.0\n",
               sigma_dense, sigma_sparse);
        return 1;
      }
    }
  }
#endif

#ifdef SINGLE
  printf("testutilsf passed.\n");
#else
  printf("testutils passed.\n");
#endif
  return 0;
}
