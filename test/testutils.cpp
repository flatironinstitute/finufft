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
#include <finufft/heuristics.hpp>  // complexity-based upsampfac (sigma) picker
#include <finufft/test_defs.hpp>
#include <finufft_common/kernel.h> // plan-time piecewise-Horner kernel fit

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
  // Plan-time piecewise-Horner kernel fit (finufft::kernel::fit_horner_coeffs). Both
  // storage precisions are explicit, so the block runs once, in the double build.
  {
    // (A) The fit stays under the aliasing tolerance its width was chosen for, or under
    // the storage round-off floor where that is larger. It sweeps sigma because beta
    // moves with sigma, so the hardest fit sits at the top of the rail, not at a round
    // sigma. Both nc bounds are guarded: one row fewer fails at either end.
    const auto check_fit = [](auto proto, double sigma, int ns) {
      using T = decltype(proto);
      finufft_spread_opts spopts{};
      spopts.nspread    = ns;
      spopts.upsampfac  = sigma;
      spopts.kerformula = 8; // PSWF with the shifted shape param, as the plans use
      set_kernel_shape_given_ns(spopts, 0);
      const int nc = max_nc_given_ns<T>(ns);
      std::vector<T> coeffs(std::size_t(nc) * ns);
      fit_horner_coeffs<T>(spopts, nc, ns, 0.0, coeffs.data());
      const auto phi   = kernel_definition_lambda(spopts);
      // The aliasing law the width comes from, tol = tolfac*exp(-(ns-1)*pi*u), inverted
      // for the tol that produced this ns. tolfac is smallest for 1D types 1 and 2 and
      // the forward formula ceils, so this is the strictest tol any plan asks.
      const double u   = std::sqrt(1.0 - 1.0 / sigma);
      const double tol = kernel_tolfac(1, 1) * std::exp(-(ns - 1) * PI * u);
      // The fit must not eat a noticeable share of the budget. Both constants are
      // empirical, with no analytic bound behind them; widen them only against a
      // measurement. Aliasing binds at small ns, the eps floor at large ns.
      const double bound =
          std::max(0.2 * tol, 48.0 * double(std::numeric_limits<T>::epsilon()));
      double worst = 0.0;
      for (int i = 0; i < ns; ++i) { // ................... loop over panels
        const double xshift = 2.0 * i + 1 - ns;
        for (int k = 0; k < 200; ++k) {
          const double z = -1.0 + (2.0 * k + 1.0) / 200.0; // panel ordinate, in (-1,1)
          const T x = T(i - 0.5 * ns + 0.5 * (z + 1.0));   // the same point, grid units
          const double got =
              double(evaluate_kernel_horner<T>(x, ns, nc, coeffs.data(), ns));
          worst = std::max(worst, std::abs(got - phi((z + xshift) / ns)));
        }
      }
      if (worst > bound)
        printf("fail: horner fit %s sigma=%.3f ns=%d nc=%d: err %.3g exceeds %.3g\n",
               sizeof(T) == 4 ? "float" : "double", sigma, ns, nc, worst, bound);
      return worst <= bound;
    };

    // (B) The fit is always double, so the float table is the rounded double table.
    // Guards against a per-precision fit, which loses two digits for nothing.
    const auto check_float_is_rounded_double = [](double sigma, int ns) {
      finufft_spread_opts spopts{};
      spopts.nspread    = ns;
      spopts.upsampfac  = sigma;
      spopts.kerformula = 8;
      set_kernel_shape_given_ns(spopts, 0);
      const int nc = max_nc_given_ns<float>(ns);
      std::vector<float> cf(std::size_t(nc) * ns);
      std::vector<double> cd(std::size_t(nc) * ns);
      fit_horner_coeffs<float>(spopts, nc, ns, 0.0, cf.data());
      fit_horner_coeffs<double>(spopts, nc, ns, 0.0, cd.data());
      for (std::size_t i = 0; i < cf.size(); ++i)
        if (cf[i] != float(cd[i])) {
          printf("fail: float horner coeff %zu differs from the rounded double fit, "
                 "sigma=%.3f ns=%d\n",
                 i, sigma, ns);
          return false;
        }
      return true;
    };

    // Sweep every sigma makeplan accepts without warning, which is wider than the
    // auto-heuristic's rail: a user may set opts.upsampfac by hand.
    for (int k = 0; k <= 39; ++k) {
      const double sigma = 1.05 + 0.05 * k; // 1.05 .. 3.00
      for (int ns = MIN_NSPREAD; ns <= MAX_NSPREAD<float>; ++ns) {
        // clamp_kernel_ns caps float to FLOAT_MAX_NS_CC below FLOAT_CC_UPSAMPFAC_LIMIT
        if (sigma < FLOAT_CC_UPSAMPFAC_LIMIT && ns > FLOAT_MAX_NS_CC) continue;
        if (!check_fit(0.0f, sigma, ns) || !check_float_is_rounded_double(sigma, ns))
          return 1;
      }
      for (int ns = MIN_NSPREAD; ns <= MAX_NSPREAD<double>; ++ns)
        if (!check_fit(0.0, sigma, ns)) return 1;
    }
  }

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
