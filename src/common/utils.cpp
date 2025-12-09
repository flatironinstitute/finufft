
#include <cmath>
#include <finufft_common/common.h>
#include <limits>

// Prefer the standard library's special-math `cyl_bessel_i` when available.
#if defined(__has_include)
#if __has_include(<version>)
#include <version>
#endif
#endif
// Feature-test macro for special math functions (if available in the standard
// library implementation). Fall back to our series implementation otherwise.
#if defined(__cpp_lib_math_special_functions)
#define FINUFFT_HAVE_STD_CYL_BESSEL_I 1
#endif

#ifdef __CUDACC__
#include <cufinufft/types.h>
#endif

namespace finufft {
namespace common {

void gaussquad(int n, double *xgl, double *wgl) {
  double x = 0, dx = 0;
  int convcount = 0;

  xgl[n / 2] = 0;                   // If odd number of nodes, middle node is 0
  for (int i = 0; i < n / 2; i++) { // Loop through nodes
    convcount = 0;
    x         = std::cos((2 * i + 1) * PI / (2 * n)); // Initial guess: Chebyshev node
    while (true) {                                    // Newton iteration
      auto [p, dp] = leg_eval(n, x);
      dx           = -p / dp;
      x += dx; // Newton step
      if (std::abs(dx) < 1e-14) {
        convcount++;
      }
      if (convcount == 3) {
        break;
      }
    }
    xgl[i]         = -x;
    xgl[n - i - 1] = x; // Symmetric nodes
  }

  for (int i = 0; i < n / 2 + 1; i++) {
    auto [junk1, dp] = leg_eval(n, xgl[i]);
    auto [p, junk2]  = leg_eval(n + 1, xgl[i]);
    wgl[i]           = -2 / ((n + 1) * dp * p);
    wgl[n - i - 1]   = wgl[i];
  }
}

std::tuple<double, double> leg_eval(int n, double x) {
  if (n == 0) {
    return {1.0, 0.0};
  }
  if (n == 1) {
    return {x, 1.0};
  }
  double p0 = 0.0, p1 = 1.0, p2 = x;
  for (int i = 1; i < n; i++) {
    p0 = p1;
    p1 = p2;
    p2 = ((2 * i + 1) * x * p1 - i * p0) / (i + 1);
  }
  return {p2, n * (x * p2 - p1) / (x * x - 1)};
}

double cyl_bessel_i(double nu, double x) noexcept {
#if defined(FINUFFT_HAVE_STD_CYL_BESSEL_I)
  return std::cyl_bessel_i(nu, x);
#else
  if (x == 0.0) {
    if (nu == 0.0) return 1.0;
    return 0.0;
  }

  const double halfx = x / 2.0;
  double term        = std::pow(halfx, nu) / std::tgamma(nu + 1.0); // k = 0
  double sum         = term;

  static constexpr auto eps      = std::numeric_limits<double>::epsilon() * 10.0;
  static constexpr auto max_iter = 100000;

  for (int k = 1; k < max_iter; ++k) {
    term *= (halfx * halfx) / (static_cast<double>(k) * (nu + static_cast<double>(k)));
    sum += term;

    if (std::abs(term) < eps * std::abs(sum)) {
      break;
    }
  }
  return sum;
#endif
}

} // namespace common
} // namespace finufft

namespace cufinufft {
namespace utils {

long next235beven(long n, long b) {
  if (n <= 2) return 2;
  if (n % 2 == 1) n += 1;                // even
  long nplus  = n - 2;                   // to cancel out the +=2 at start of loop
  long numdiv = 2;                       // a dummy that is >1
  while ((numdiv > 1) || (nplus % b != 0)) {
    nplus += 2;                          // stays even
    numdiv = nplus;
    while (numdiv % 2 == 0) numdiv /= 2; // remove all factors of 2,3,5...
    while (numdiv % 3 == 0) numdiv /= 3;
    while (numdiv % 5 == 0) numdiv /= 5;
  }
  return nplus;
}

} // namespace utils
} // namespace cufinufft
