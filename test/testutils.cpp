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
using namespace finufft::kernel;

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

    // ns is non-increasing as sigma rises (the minimizer enumerates one candidate
    // per achievable width). Double holds over the whole auto range. Float holds only
    // where the cancellation guard in clamp_kernel_ns (kernel.h) does not cap ns: the
    // cap rises with sigma, so a capped ns can step up.
    const double tols[] = {1e-3, 1e-6, 1e-10, 1e-13};
    for (int dim = 1; dim <= 3; ++dim)
      for (double tol : tols) {
        int prev_d = 1 << 30, prev_f = 1 << 30;
        for (int i = 0; i <= 25; ++i) { // step 0.05; an index, as a running sum drifts
          const double s =
              MIN_AUTO_UPSAMPFAC + (MAX_AUTO_UPSAMPFAC - MIN_AUTO_UPSAMPFAC) * i / 25;
          const int nd = kernel_width_at<double>(tol, dim, s);
          if (nd > prev_d) {
            printf("fail: ns(double) rose: dim=%d tol=%.0e sigma=%.2f\n", dim, tol, s);
            return 1;
          }
          prev_d       = nd;
          const int nf = kernel_width_at<float>(tol, dim, s);
          // skip float sigmas where the cancellation guard caps ns (low-sigma prefix)
          if (nf < std::min(ns_for_sigma(tol, dim, s), ns_f)) continue;
          if (nf > prev_f) {
            printf("fail: ns(float) rose: dim=%d tol=%.0e sigma=%.2f\n", dim, tol, s);
            return 1;
          }
          prev_f = nf;
        }
      }

    // The inverses (kernel.h) round-trip with their forward functions: each returns the
    // smallest 3-digit sigma above 1 that meets its target, or infinity if
    // MAX_AUTO_UPSAMPFAC does not meet it.
    {
      const auto round_trips = [&](double s, const auto &ok) {
        if (!std::isfinite(s)) return !ok(MAX_AUTO_UPSAMPFAC);
        const double milli = std::round(1000 * s);
        return milli / 1000 == s && ok(s) && (milli == 1001 || !ok((milli - 1) / 1000));
      };
      int checked = 0;
      for (int dim = 1; dim <= 3; ++dim)
        for (int ns = 2; ns <= 16; ++ns)
          for (int k = 0; k <= 15; ++k) {
            const double tol   = k == 0 ? 0.9 : std::pow(10.0, -k);
            const auto gets_ns = [&](double s) {
              return ns_for_sigma(tol, dim, s) <= ns;
            };
            if (!round_trips(sigma_for_ns(tol, dim, ns), gets_ns)) {
              printf("fail: sigma_for_ns does not round-trip with ns_for_sigma: dim=%d "
                     "ns=%d tol=%.0e\n",
                     dim, ns, tol);
              return 1;
            }
            ++checked;
            for (double eps : {eps_d, eps_f}) {
              const auto ok = [&](double s) {
                return deconv_error(eps, dim, ns, s, alias_decay(s, 0.5 / s)) <= tol;
              };
              if (!round_trips(sigma_for_deconv(eps, dim, ns, tol), ok)) {
                printf("fail: sigma_for_deconv does not round-trip with deconv_error: "
                       "dim=%d ns=%d eps=%.1e tol=%.0e\n",
                       dim, ns, eps, tol);
                return 1;
              }
              ++checked;
            }
          }
      // Controls, at an edge the loop proved: one 3-digit step below and above it, and
      // one ulp above it.
      const double s8 = sigma_for_ns(1e-6, 1, 8), m8 = std::round(1000 * s8);
      const auto ok8 = [&](double x) {
        return ns_for_sigma(1e-6, 1, x) <= 8;
      };
      if (round_trips((m8 - 1) / 1000, ok8) || round_trips((m8 + 1) / 1000, ok8) ||
          round_trips(std::nextafter(s8, 2.0), ok8)) {
        printf("fail: round_trips accepts a sigma off the 3-digit edge\n");
        return 1;
      }
      printf("sigma_for_ns and sigma_for_deconv round-trip with their forward functions "
             "in %d cells; the three off-edge controls fail\n",
             checked);
    }

    // Every auto pick other than the fallback MAX_AUTO_UPSAMPFAC is feasible, and the
    // fallback is not: a feasible fallback is a width edge that minimize missed.
    // Control: in the 2D N=16 tol 5.6234e-14 cell only an interior sigma is feasible,
    // so a pick stuck at either end of the auto range fails.
    {
      int picks = 0, fallbacks = 0, bad = 0;
      const auto check = [&](auto prec, int dim, int type, double N, double tol) {
        using TF           = decltype(prec);
        const double nm[3] = {N, N, N}, X[3] = {3, 3, 3}, S[3] = {N, N, N};
        const double sigma = type == 3
                                 ? best_type3<TF>(tol, dim, 1, 1e3, X, S, 1e3)
                                 : best_type12<TF>(tol, dim, type, 1, nm, 1e3).sigma;
        const bool ok      = upsampfac_feasible(
            sigma, tol, dim, type, std::is_same_v<TF, float>, type == 3 ? 1.0 : N);
        ++picks;
        fallbacks += !ok && sigma == MAX_AUTO_UPSAMPFAC;
        bad += ok == (sigma == MAX_AUTO_UPSAMPFAC);
        return ok;
      };
      const auto feasible_16 = [&](double s) {
        return upsampfac_feasible(s, 5.6234e-14, 2, 1, false, 16);
      };
      if (feasible_16(MIN_AUTO_UPSAMPFAC) || feasible_16(MAX_AUTO_UPSAMPFAC) ||
          !check(0.0, 2, 1, 16, 5.6234e-14)) {
        printf("fail: control cell: the pick is not the interior feasible sigma\n");
        return 1;
      }
      for (int dim = 1; dim <= 3; ++dim)
        for (int type = 1; type <= 3; ++type)
          for (double N : {16.0, 256.0, 1e5})
            for (int k = 2; k <= 14; ++k) {
              check(0.0, dim, type, N, std::pow(10.0, -k));
              check(0.0f, dim, type, N, std::pow(10.0, -k));
            }
      printf("%d auto picks: %d feasible, %d fallbacks at MAX_AUTO_UPSAMPFAC, %d "
             "infeasible or feasible fallbacks; the interior-only control cell passes\n",
             picks, picks - fallbacks - bad, fallbacks, bad);
      if (bad > 0 || picks == fallbacks) {
        printf(
            "fail: an infeasible pick below MAX_AUTO_UPSAMPFAC, a feasible fallback, or "
            "no feasible pick\n");
        return 1;
      }
    }

    // Density drives the pick: a spread-dominated transform (many points, small
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
      if (!(sigma_dense > sigma_sparse) || !(sigma_dense > 2.0)) {
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
