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

namespace finufft::common {
double cyl_bessel_i_custom(double nu, double x) noexcept;
} // namespace finufft::common

using namespace finufft::common;
using namespace finufft::heuristics;

int main(int argc, char *argv[]) {
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

#if defined(__cpp_lib_math_special_functions)
  // std::cyl_bessel_i present: compare std vs custom series
  for (double x = 0.0; x <= 42.0; x += 0.5) {
    double stdv    = std::cyl_bessel_i(0, x);
    double custom  = finufft::common::cyl_bessel_i_custom(0, x);
    double rel_err = std::abs(1.0 - stdv / custom);
    if (rel_err > std::numeric_limits<double>::epsilon() * 20) {
      printf("fail: Bessel mismatch at x=%g: std=%g custom=%g rel_err=%g\n", x, stdv,
             custom, rel_err);
      return 1;
    }
  }
#else
  printf("Bessel comparison test skipped. std bessel function not available.\n");
#endif

#ifndef SINGLE
  // Complexity-based upsampfac (sigma) picker (finufft/heuristics.hpp). The block
  // exercises both precisions explicitly, so it runs once in the double build.
  {
    const double eps_d = std::numeric_limits<double>::epsilon();
    const double eps_f = std::numeric_limits<float>::epsilon();
    const int ns_d = MAX_NSPREAD<double>, ns_f = MAX_NSPREAD<float>;

    // (A) ns is non-increasing as sigma rises (the minimizer enumerates one candidate
    // per achievable width). Double holds over the whole auto range; float only above
    // FLOAT_CC_UPSAMPFAC_LIMIT, since below it the catastrophic-cancellation guard caps
    // ns low, so ns jumps up at the threshold.
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
            prev_d = nd;
            if (s < FLOAT_CC_UPSAMPFAC_LIMIT) continue; // skip float CC-capped region
            const int nf = kernel_width_at<float>(tol, dim, type, s);
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

    // (C) sigma=2.5 is feasible down to eps_mach for every dim/type, both precisions ->
    // analytic_upsampfac never returns an infeasible sigma for any tol the pipeline
    // forwards (it clamps tol up to eps_mach first).
    for (int dim = 1; dim <= 3; ++dim)
      for (int type = 1; type <= 3; ++type) {
        const double maxN = 256;
        if (!upsampfac_feasible(MAX_AUTO_UPSAMPFAC, eps_d, dim, type, eps_d, ns_d, false,
                                maxN) ||
            !upsampfac_feasible(MAX_AUTO_UPSAMPFAC, eps_f, dim, type, eps_f, ns_f, true,
                                maxN)) {
          printf("fail: sigma=2.5 infeasible at eps_mach: dim=%d type=%d\n", dim, type);
          return 1;
        }
      }

    // (D) analytic_upsampfac returns a sigma that is itself feasible, for a range of
    // achievable tols (its contract: the pick always survives the real plan).
    for (double tol : tols) {
      const double maxN = 1e4;
      const double s = analytic_upsampfac(tol, 2, 1, eps_d, ns_d, false, maxN);
      if (!(s >= MIN_AUTO_UPSAMPFAC - 1e-9 && s <= MAX_AUTO_UPSAMPFAC + 1e-9) ||
          !upsampfac_feasible(s, tol, 2, 1, eps_d, ns_d, false, maxN)) {
        printf("fail: analytic sigma %.3f not feasible/in range at tol=%.0e\n", s, tol);
        return 1;
      }
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
      if (!(sigma_dense > sigma_sparse) || !(sigma_dense > MAX_CHECK_SIGMA - 1e-9)) {
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
