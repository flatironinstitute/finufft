#pragma once

#include <cmath>
#include <complex>
#ifdef __CUDACC__
#include <thrust/complex.h>
#endif

// ------------------------------------------------------------
// 1D type‑1 NUFFT, direct:
//   f[m] = Σ_{j=0..nj-1} c[j] exp(i*iflag * k_m * x[j])
//   k_m = -⌊ms/2⌋ + m
// ------------------------------------------------------------
template<typename BIGINT,
         typename XArr, // xArr[j] yields real FLT
         typename CArr, // cArr[j] yields Complex<FLT>
         typename FArr  // fArr[m] yields Complex<FLT>
         >
void dirft1d1(const BIGINT nj,
              const XArr &xArr,
              const CArr &cArr,
              const int iflag,
              const BIGINT ms,
              FArr &fArr) {
  using Complex = std::decay_t<decltype(cArr[0])>;
  using FLT     = typename Complex::value_type;

  static_assert(std::is_floating_point<FLT>::value,
                "dirft1d1: Complex::value_type must be a floating point");
  static_assert(std::is_same<std::decay_t<decltype(xArr[0])>, FLT>::value,
                "dirft1d1: xArr element type must match Complex::value_type");

  BIGINT kmin = -(ms / 2);
  Complex I0{0, FLT(iflag)};

  for (BIGINT m = 0; m < ms; ++m) fArr[m] = Complex{0, 0};

  for (BIGINT j = 0; j < nj; ++j) {
    FLT xj        = xArr[j];
    Complex phase = exp(I0 * xj);
    Complex p     = pow(phase, static_cast<FLT>(kmin));
    Complex cj    = cArr[j];
    for (BIGINT m = 0; m < ms; ++m) {
      fArr[m] += cj * p;
      p *= phase;
    }
  }
}

// ------------------------------------------------------------
// 1D type‑2 NUFFT, direct:
//   c[j] = Σ_{m=0..ms-1} f[m] exp(i*iflag * k_m * x[j])
// ------------------------------------------------------------
template<typename BIGINT,
         typename XArr, // xArr[j] yields real FLT
         typename CArr, // cArr[j] yields Complex<FLT>
         typename FArr  // fArr[m] yields Complex<FLT>
         >
void dirft1d2(const BIGINT nj,
              const XArr &xArr,
              CArr &cArr,
              const int iflag,
              const BIGINT ms,
              const FArr &fArr) {
  using Complex = std::decay_t<decltype(cArr[0])>;
  using FLT     = typename Complex::value_type;

  static_assert(std::is_floating_point<FLT>::value,
                "dirft1d2: Complex::value_type must be a floating point");
  static_assert(std::is_same<std::decay_t<decltype(xArr[0])>, FLT>::value,
                "dirft1d2: xArr element type must match Complex::value_type");

  BIGINT kmin = -(ms / 2);
  Complex I0{0, FLT(iflag)};

  for (BIGINT j = 0; j < nj; ++j) {
    FLT xj        = xArr[j];
    Complex phase = exp(I0 * xj);
    Complex p     = pow(phase, static_cast<FLT>(kmin));
    Complex sum{0, 0};
    for (BIGINT m = 0; m < ms; ++m) {
      sum += fArr[m] * p;
      p *= phase;
    }
    cArr[j] = sum;
  }
}

// ------------------------------------------------------------
// 1D type‑3 NUFFT, direct:
//   f[k] = Σ_{j=0..nj-1} c[j] exp(i*iflag * s[k] * x[j])
// ------------------------------------------------------------
template<typename BIGINT,
         typename XArr, // xArr[j] yields real FLT
         typename CArr, // cArr[j] yields Complex<FLT>
         typename SArr, // sArr[k] yields real FLT
         typename FArr  // fArr[k] yields Complex<FLT>
         >
void dirft1d3(const BIGINT nj,
              const XArr &xArr,
              const CArr &cArr,
              const int iflag,
              const BIGINT nk,
              const SArr &sArr,
              FArr &fArr) {
  using Complex = std::decay_t<decltype(cArr[0])>;
  using FLT     = typename Complex::value_type;

  static_assert(std::is_floating_point<FLT>::value,
                "dirft1d3: Complex::value_type must be a floating point");
  static_assert(std::is_same<std::decay_t<decltype(xArr[0])>, FLT>::value,
                "dirft1d3: xArr element type must match Complex::value_type");
  static_assert(std::is_same<std::decay_t<decltype(sArr[0])>, FLT>::value,
                "dirft1d3: sArr element type must match Complex::value_type");

  Complex I0{0, FLT(iflag)};

  for (BIGINT k = 0; k < nk; ++k) {
    Complex ss = I0 * sArr[k];
    Complex sum{0, 0};
    for (BIGINT j = 0; j < nj; ++j) sum += cArr[j] * exp(ss * xArr[j]);
    fArr[k] = sum;
  }
}
