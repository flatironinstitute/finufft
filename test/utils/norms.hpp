#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <type_traits>
#ifdef __CUDACC__
#include <thrust/complex.h>
#endif

// ----------------------------------------------------------------------------
// Compute ||a–b||₂ / ||a||₂
// ----------------------------------------------------------------------------
template<typename BIGINT,
         typename ArrA, // supports ArrA[i] -> complex-like
         typename ArrB  // supports ArrB[i] -> same complex-like
         >
auto relerrtwonorm(BIGINT n, ArrA aArr, ArrB bArr) -> decltype(aArr[0].real()) {
  using Complex = std::decay_t<decltype(aArr[0])>;
  using FLT     = decltype(aArr[0].real());

  static_assert(std::is_floating_point<FLT>::value,
                "relerrtwonorm: value_type must be floating point");
  static_assert(std::is_same<Complex, std::decay_t<decltype(bArr[0])>>::value,
                "relerrtwonorm: array element types must match");

  FLT err = FLT{0}, nrm = FLT{0};
  for (BIGINT i = 0; i < n; ++i) {
    auto ai = aArr[i], bi = bArr[i];
    FLT dr = ai.real() - bi.real(), di = ai.imag() - bi.imag();
    err += dr * dr + di * di;
    FLT ar = ai.real(), ai_im = ai.imag();
    nrm += ar * ar + ai_im * ai_im;
  }
  return std::sqrt(err / nrm);
}

// ----------------------------------------------------------------------------
// Compute ||a–b||₂
// ----------------------------------------------------------------------------
template<typename BIGINT, typename ArrA, typename ArrB>
auto errtwonorm(BIGINT n, ArrA aArr, ArrB bArr) -> decltype(aArr[0].real()) {
  using Complex = std::decay_t<decltype(aArr[0])>;
  using FLT     = decltype(aArr[0].real());

  static_assert(std::is_floating_point<FLT>::value,
                "errtwonorm: value_type must be floating point");
  static_assert(std::is_same<Complex, std::decay_t<decltype(bArr[0])>>::value,
                "errtwonorm: array element types must match");

  FLT err = FLT{0};
  for (BIGINT i = 0; i < n; ++i) {
    auto ai = aArr[i], bi = bArr[i];
    FLT dr = ai.real() - bi.real(), di = ai.imag() - bi.imag();
    err += dr * dr + di * di;
  }
  return std::sqrt(err);
}

// ----------------------------------------------------------------------------
// Compute ||a||₂
// ----------------------------------------------------------------------------
template<typename BIGINT, typename ArrA>
auto twonorm(BIGINT n, ArrA aArr) -> decltype(aArr[0].real()) {
  using FLT = decltype(aArr[0].real());

  static_assert(std::is_floating_point<FLT>::value,
                "twonorm: value_type must be floating point");

  FLT nrm = FLT{0};
  for (BIGINT i = 0; i < n; ++i) {
    auto ai = aArr[i];
    FLT ar = ai.real(), ai_im = ai.imag();
    nrm += ar * ar + ai_im * ai_im;
  }
  return std::sqrt(nrm);
}

// ----------------------------------------------------------------------------
// Compute ||a||_∞
// ----------------------------------------------------------------------------
template<typename BIGINT, typename ArrA>
auto infnorm(BIGINT n, ArrA aArr) -> decltype(aArr[0].real()) {
  using FLT = decltype(aArr[0].real());

  static_assert(std::is_floating_point<FLT>::value,
                "infnorm: value_type must be floating point");

  FLT maxv = FLT{0};
  for (BIGINT i = 0; i < n; ++i) {
    auto ai = aArr[i];
    FLT ar = ai.real(), ai_im = ai.imag();
    FLT mag2 = ar * ar + ai_im * ai_im;
    maxv     = std::max(maxv, mag2);
  }
  return std::sqrt(maxv);
}
