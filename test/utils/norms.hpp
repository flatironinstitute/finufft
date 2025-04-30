#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <type_traits>
#ifdef __CUDACC__
#include <thrust/complex.h>
#endif

// NOTE: not using std::func but just func for thrust::complex compatibility
// AHB utilities, MB templated them.

// ----------------------------------------------------------------------------
// Compute ||a–b||₂ / ||a||₂
// ----------------------------------------------------------------------------
template<typename BIGINT,
         typename ArrA, // supports ArrA[i] -> complex-like
         typename ArrB  // supports ArrB[i] -> same complex-like
         >
auto relerrtwonorm(BIGINT n, ArrA a, ArrB b) -> decltype(a[0].real()) {
  using Complex = std::decay_t<decltype(a[0])>;
  using FLT     = decltype(a[0].real());

  static_assert(std::is_floating_point_v<FLT>,
                "relerrtwonorm: value_type must be floating point");
  static_assert(std::is_same_v<Complex, std::decay_t<decltype(b[0])>>,
                "relerrtwonorm: array element types must match");

  FLT err = 0.0, nrm = 0.0;
  for (BIGINT m = 0; m < n; ++m) {
    // note std::norm here & below is |a|^2 ("field norm") not usual |a| ...
    nrm += norm(a[m]);
    err += norm(a[m] - b[m]);
  }
  return sqrt(err / nrm);
}

// ----------------------------------------------------------------------------
// Compute ||a–b||₂
// ----------------------------------------------------------------------------
template<typename BIGINT, typename ArrA, typename ArrB>
auto errtwonorm(BIGINT n, ArrA a, ArrB b) -> decltype(a[0].real()) {
  using Complex = std::decay_t<decltype(a[0])>;
  using FLT     = decltype(a[0].real());

  static_assert(std::is_floating_point_v<FLT>,
                "errtwonorm: value_type must be floating point");
  static_assert(std::is_same_v<Complex, std::decay_t<decltype(b[0])>>,
                "errtwonorm: array element types must match");

  FLT err = 0.0; // compute error 2-norm
  for (BIGINT m = 0; m < n; ++m) err += norm(a[m] - b[m]);
  return sqrt(err);
}

// ----------------------------------------------------------------------------
// Compute ||a||₂
// ----------------------------------------------------------------------------
template<typename BIGINT, typename ArrA>
auto twonorm(BIGINT n, ArrA a) -> decltype(a[0].real()) {
  using FLT = decltype(a[0].real());

  static_assert(std::is_floating_point_v<FLT>,
                "twonorm: value_type must be floating point");

  FLT nrm = 0.0;
  for (BIGINT m = 0; m < n; ++m) nrm += norm(a[m]);
  return sqrt(nrm);
}

// ----------------------------------------------------------------------------
// Compute ||a||_∞
// ----------------------------------------------------------------------------
template<typename BIGINT, typename ArrA>
auto infnorm(BIGINT n, ArrA a) -> decltype(a[0].real()) {
  using FLT = decltype(a[0].real());

  static_assert(std::is_floating_point_v<FLT>,
                "infnorm: value_type must be floating point");

  FLT nrm = 0.0;
  for (BIGINT m = 0; m < n; ++m) nrm = std::max(nrm, norm(a[m]));
  return sqrt(nrm);
}
