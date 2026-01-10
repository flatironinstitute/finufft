#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include <finufft_common/constants.h>
#include <finufft_common/utils.h>
#include <finufft_errors.h>
#include <finufft_spread_opts.h>

namespace finufft::kernel {

template<class T, class F> std::vector<T> poly_fit(F &&f, int n, T dummy) noexcept {
  static_assert(std::is_floating_point_v<T>, "T must be floating-point");
  /* Expects f, a function handle for arguments on [-1,1], both I/O type T.
     Returns vector of n coefficients a_{n-1}, ... a_1, a_0 of degree-(n-1)
     polynomial that interpolates f at a set of hard-wired Chebychev nodes.

     Barbone, fall 2025. Barnett removed a,b to reduce poly defn confusion 12/29/25.
     dummy seems to be needed to communicate the T type.
  */

  // 1) Type-1 Chebyshev nodes t_k, data samples y_k = f(t_k)
  std::vector<T> t(n), y(n);
  for (int k = 0; k < n; ++k) {
    t[k]       = std::cos((T(2 * k + 1) * common::PI) / (T(2) * T(n))); // in (-1,1)
    // t[k]       = std::cos((T(k) * common::PI) / T(n-1)); // type-2 in [-1,1] also ok
    y[k] = static_cast<T>(f(t[k])); // evaluate this sample
  }

  // 2) Newton divided differences on t: coef[j] = f[t0,..,tj]
  std::vector<T> coef = y;
  for (int j = 1; j < n; ++j)
    for (int i = n - 1; i >= j; --i)
      coef[i] = (coef[i] - coef[i - 1]) / (t[i] - t[i - j]);

  // 3) Convert Newton form to monomial coeffs in t (low→high)
  // Multiply a polynomial p(t) by (t - c).
  // Input p holds coefficients low->high (p[0] + p[1] t + p[2] t^2 + ...).
  // Returns r of length p.size()+1 so that r represents (t - c)*p(t).
  // Concretely: r[i] = -c*p[i] + (i>0 ? p[i-1] : 0) for i=0..p.size()-1,
  // and r[p.size()] = p[p.size()-1].
  auto mul_by_linear = [](const std::vector<T> &p, T c) {
    std::vector<T> r(p.size() + 1, T(0));
    for (std::size_t i = 0; i < p.size(); ++i) {
      r[i] += -c * p[i]; // (t - c)*p
      r[i + 1] += p[i];
    }
    return r;
  };

  std::vector<T> c(n, T(0));  // output coefficients in t
  std::vector<T> basis{T(1)}; // Π_{m<j} (t - t_m), low→high
  c[0] += coef[0];
  for (int j = 1; j < n; ++j) {
    basis = mul_by_linear(basis, t[j - 1]);
    for (std::size_t m = 0; m < basis.size(); ++m) c[m] += coef[j] * basis[m];
  }
  std::reverse(c.begin(), c.end()); // convert to Horner ordering (high -> low)
  return c;
}

template<typename T> T kernel_definition(T x, T beta, T c, int kerformula) {
  /* The spread/interp kernel function definitions.
     The single real argument x is in (-ns/2, ns/2), where ns
     is the integer spreading width (support in fine gridpoints).

     *** currently ns not known, encoded in c -> change so halfwidth passsed in.
     *** to kill c and have this kernel being defined on [-1,1]

     kerformula must be >0 (even though 0 indicates default; this must have
     already happened).

     kerformula=1 :  ES ("exp sqrt") kernel (default)
     phi_ES(x) = exp(beta*(sqrt(1 - c*x^2) - 1))
     Note: max is 1. The kernel used in FINUFFT 2017-2025.

     kerformula=2 : Kaiser--Bessel (KB) kernel
       phi_KB(x) = I_0(beta*sqrt(1 - c*x^2)) / I_0(beta)
     Note: `std::cyl_bessel_i` from <cmath> is used for I_0.
     Note: max is 1.
  */
  if (kerformula == 1) {
    if (c * x * x >= T(1)) return T(0.0); // prevent nonpositive sqrts
    return std::exp(beta * (std::sqrt(T(1) - c * x * x) - T(1))); // ES formula

  } else if (kerformula == 2) {
    // Kaiser--Bessel (normalized by I0(beta)). Use std::cyl_bessel_i from <cmath>.
    if (c * x * x >= T(1)) return T(0.0); // prevent nonpositive sqrts
    const T inner        = std::sqrt(T(1) - c * x * x);
    const T arg          = beta * inner;
    const double i0_arg  = ::finufft::common::cyl_bessel_i(0, static_cast<double>(arg));
    const double i0_beta = ::finufft::common::cyl_bessel_i(0, static_cast<double>(beta));
    return static_cast<T>(i0_arg / i0_beta);

  } else {
    fprintf(stderr, "[%s] unknown opts.kerformula=%d\n", __func__, kerformula);
    throw int(FINUFFT_ERR_KERFORMULA_NOTVALID);
    return T(0.0);
  }
}

FINUFFT_EXPORT int theoretical_kernel_ns(double tol, int dim, int type, int debug,
                      const finufft_spread_opts &spopts);

FINUFFT_EXPORT void set_kernel_shape_given_ns(finufft_spread_opts &opts);

template<int NS, int NC> inline constexpr bool ValidKernelParams() noexcept {
  // NS = nspread (kernel width), NC = # poly coeffs in Horner evaluator.
  // Defines the compiled range of NC for each NS, in spreadinterp.
  // Other instantiations can be
  // compiled away at call sites using if constexpr to reduce binary size.
  // Barbone Dec 2025.
  return (NC >= NS - 1) && (NC <= NS + 3);
}

} // namespace finufft::kernel
