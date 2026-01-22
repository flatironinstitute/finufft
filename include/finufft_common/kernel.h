#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include <finufft_common/constants.h>
#include <finufft_common/pswf.h>
#include <finufft_common/spread_opts.h>
#include <finufft_common/utils.h>
#include <finufft_errors.h>

namespace finufft::kernel {

template<class T, class F> std::vector<T> poly_fit(F &&f, int n) noexcept {
  static_assert(std::is_floating_point_v<T>, "T must be floating-point");
  /* Expects f, a function handle for arguments on [-1,1], both I/O type T.
     Returns vector of n coefficients a_{n-1}, ... a_1, a_0 of degree-(n-1)
     polynomial that interpolates f at a set of hard-wired Chebyshev nodes.

     Barbone, Fall 2025.
     Barnett 12/29/25-1/13/26 removed a,b to simplify; no poly defn confusion.
  */

  // 1) Type-1 Chebyshev nodes t_k, data samples y_k = f(t_k)
  std::vector<T> t(n), y(n);
  for (int k = 0; k < n; ++k) {
    t[k] = std::cos((T(2 * k + 1) * common::PI) / (T(2) * T(n))); // in (-1,1)
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

inline double kernel_definition(const finufft_spread_opts &spopts, const double z) {
  /* The spread/interp kernel phi_beta(z) function on standard interval z in [-1,1],
     This evaluation does not need to be fast; it is used *only* for polynomial
     interpolation via Horner coeffs (the interpolant is evaluated fast).
     It can thus always be double-precision. No analytic Fourier transform pair is
     needed, thanks to numerical quadrature in finufft_core:onedim*; playing with
     new kernels is thus very easy.
    Inputs:
    z      - real ordinate on standard interval [-1,1]. Handling of edge cases
            at or near +-1 is no longer crucial, because precompute_horner_coeffs
            (the only user of this function) has interpolation nodes in (-1,1).
    spopts - spread_opts struct containing fields:
      beta        - shape parameter for ES, KB, or other prolate kernels
                    (a.k.a. c parameter in PSWF).
      kerformula  - positive integer selecting among kernel function types; see
                    docs in the code below.
                    (More than one value may give the same type, to allow
                    kerformula also to select a parameter-choice method.)
                    Note: the default 0 (in opts.spread_kerformula) is invalid here;
                    selection of a >0 kernel type must already have happened.
    Output: phi(z), as in the notation of original 2019 paper ([FIN] in the docs).

    Notes: 1) no normalization of max value or integral is needed, since any
            overall factor is cancelled out in the deconvolve step. However,
            values as large as exp(beta) have caused floating-pt overflow; don't
            use them.
    Barnett rewritten 1/13/26 for double on [-1,1]; based on Barbone Dec 2025.
  */
  if (std::abs(z) > 1.0) return 0.0;           // restrict support to [-1,1]
  double beta = spopts.beta;                   // get shape param
  double arg  = beta * std::sqrt(1.0 - z * z); // common argument for exp, I0, etc
  int kf      = spopts.kerformula;

  if (kf == 1 || kf == 2)
    // ES ("exponential of semicircle" or "exp sqrt"), see [FIN] reference.
    // Used in FINUFFT 2017-2025 (up to v2.4.1). max is 1, as of v2.3.0.
    return std::exp(arg) / std::exp(beta);
  else if (kf == 3)
    // forwards Kaiser--Bessel (KB), normalized to max of 1.
    // std::cyl_bessel_i is from <cmath>, expects double. See src/common/utils.cpp
    return common::cyl_bessel_i(0, arg) / common::cyl_bessel_i(0, beta);
  else if (kf == 4)
    // continuous (deplinthed) KB, as in Barnett SIREV 2022, normalized to max nearly 1
    return (common::cyl_bessel_i(0, arg) - 1.0) / common::cyl_bessel_i(0, beta);
  else if (kf == 5)
    return std::cosh(arg) / std::cosh(beta); // normalized cosh-type of Rmk. 13 [FIN]
  else if (kf == 6)
    return (std::cosh(arg) - 1.0) / std::cosh(beta); // Potts-Tasche cont cosh-type
  else if (kf == 7)
    return common::pswf(beta, z); // pswf order zero, normalized to 1 at z=0
  else {
    fprintf(stderr, "[%s] unknown spopts.kerformula=%d\n", __func__, spopts.kerformula);
    throw int(FINUFFT_ERR_KERFORMULA_NOTVALID);      // *** crashes matlab, not good
    return std::numeric_limits<double>::quiet_NaN(); // never gets here, non-signalling
  }
}

FINUFFT_EXPORT int theoretical_kernel_ns(double tol, int dim, int type, int debug,
                                         const finufft_spread_opts &spopts);

FINUFFT_EXPORT void set_kernel_shape_given_ns(finufft_spread_opts &opts, int debug);

// min and max number of poly coeffs allowed (compiled) for a given spread width ns.
// Since for low upsampfacs, ns=16 can need only nc~12, allow such low nc here.
// Note: spreadinterp.cpp compilation time grows with the gap between these bounds...
inline constexpr int min_nc_given_ns(int ns) {
  return std::max(common::MIN_NC, ns - 4); // note must stay in bounds from constants.h
}
inline constexpr int max_nc_given_ns(int ns) {
  return std::min(common::MAX_NC, ns + 3); // "
}

template<int NS, int NC> inline constexpr bool ValidKernelParams() noexcept {
  // NS = nspread (kernel width), NC = # poly coeffs in Horner evaluator.
  // Defines the compiled range of NC for each NS, in spreadinterp.
  // Other instantiations can be
  // compiled away at call sites using if constexpr to reduce binary size.
  // Barbone Dec 2025.
  // AHB changed to use the above two expressions, but needs checking if compile-time ok
  return (NC >= min_nc_given_ns(NS)) && (NC <= max_nc_given_ns(NS));
}

} // namespace finufft::kernel
