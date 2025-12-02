#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <finufft_common/constants.h>

namespace finufft::kernel {

template<class T, class F> std::vector<T> fit_monomials(F &&f, int n, T a, T b) noexcept {
  static_assert(std::is_floating_point_v<T>, "T must be floating-point");

  // map t∈[-1,1] ↔ x∈[a,b]
  const T mid  = T(0.5) * (a + b);
  const T half = T(0.5) * (b - a);

  // 1) Chebyshev nodes t_k and mapped x_k, sample y_k = f(x_k)
  std::vector<T> t(n), y(n);
  for (int k = 0; k < n; ++k) {
    t[k]       = std::cos((T(2 * k + 1) * common::PI) / (T(2) * T(n))); // in (-1,1)
    const T xk = mid + half * t[k];
    y[k]       = static_cast<T>(f(xk));
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
  std::reverse(c.begin(), c.end());
  return c;
}

template<typename T> T evaluate_kernel(T x, T beta, T c) {
  /* ES ("exp sqrt") kernel evaluation at single real argument:
      phi(x) = exp(beta.(sqrt(1 - (2x/n_s)^2) - 1)),    for |x| < nspread/2
     related to an asymptotic approximation to the Kaiser--Bessel, itself an
     approximation to prolate spheroidal wavefunction (PSWF) of order 0.
     This is the "reference implementation", used by eg finufft/onedim_* 2/17/17.
     Rescaled so max is 1, Barnett 7/21/24
  */
  return std::exp(beta * (std::sqrt(T(1) - c * x * x) - T(1)));
}

} // namespace finufft::kernel
