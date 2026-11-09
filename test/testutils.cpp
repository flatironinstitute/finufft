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
#include <finufft/test_defs.hpp>
#include <finufft_common/kernel.h>

using namespace finufft::common;
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
  if (std::abs(twonorm(M, &a[0]) - std::sqrt((FLT)M)) > relerr * std::sqrt((FLT)M))
    return 1;
  b[0] = CPX(0.0, 0.0); // perturb b from a
  if (std::abs(errtwonorm(M, &a[0], &b[0]) - 1.0) > relerr) return 1;
  if (std::abs(std::sqrt((FLT)M) * relerrtwonorm(M, &a[0], &b[0]) - 1.0) > relerr)
    return 1;

  // Test that ns -> sigma -> sigma_pad -> ns is the identity operation, where the
  // first step is min_sigma_for_aliasing_error(), the middle increases sigma slightly
  // (the sign here relies on ceil not floor being used in theoretical_kernel_ns),
  // and the third step is theoretical_kernel_ns() itself. Ie, this tests that
  // min_sigma_for_aliasing_error is the inverse function of the formula inside
  // theoretical_kernel_ns (omitting its ceil step), with respect to ns and sigma,
  // holding tol and all else fixed. This is needed so an auto-chosen sigma causes the
  // same ns as assumed in the CPU cost model to be chosen later in the plan.
  // Barnett 11/8/26 simplified and covers usage precisely needed in best_sigma_for_tol()
  {
    int nchecked = 0, nfailed = 0;
    for (int dim = 1; dim <= 3; ++dim)
      for (int k = 1; k <= 15; ++k) {
        const double tol = std::pow(10.0, -k);
        for (int ns = 2; ns <= MAX_NSPREAD<FLT>; ++ns) {
          for (int type = 1; type <= 3; ++type) {
            const auto sigma_ns = min_sigma_for_aliasing_error(tol, dim, type, ns);
            // note: since min_sigma_for_deconv_error is not invokved, the following
            // actually tests some sigma's not needed in best_sigma_for_tol...
            if (sigma_ns < MIN_AUTO_UPSAMPFAC || sigma_ns > MAX_AUTO_UPSAMPFAC) continue;
            const double sigma_pad = sigma_ns + UPSAMPFAC_EDGE_PAD; // bump, constants.h
            ++nchecked;
            int tkns = theoretical_kernel_ns(tol, dim, type, sigma_pad);
            if (tkns != ns) {
              printf("\tfail: ns=%d tkns=%d sigma_pad=%.16g\n", ns, tkns, sigma_pad);
              ++nfailed;
            }
          }
        }
      }
    if (nchecked == 0 || nfailed) { // an empty sweep proves nothing
      printf("fail: min_sigma_for_aliasing_error does not match ns in %d of %d cases\n",
             nfailed, nchecked);
      return 1;
    }
  }

#ifdef SINGLE
  printf("testutilsf passed.\n");
#else
  printf("testutils passed.\n");
#endif
  return 0;
}
