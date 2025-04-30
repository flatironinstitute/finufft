#pragma once

#include <cmath>
#include <complex>
#ifdef __CUDACC__
#include <thrust/complex.h>
#endif

// This is basically a port of dirft1d.f from CMCL package, except with
// the 1/nj prefactors for type-1 removed.

// Direct computation of 1D type-1 nonuniform FFT. Interface same as finufft1d1.
//   f[m] = Σ_{j=0..nj-1} c[j] exp(i*iflag * k_m * x[j])
//   k_m = -⌊ms/2⌋ + m
// The output array is in increasing k1 ordering. If iflag>0 the + sign is
// used, otherwise the - sign is used, in the exponential.
// Uses winding trick.  Barnett 1/25/17
template<typename BIGINT,
         typename XArr, // xArr[j] yields real FLT
         typename CArr, // cArr[j] yields Complex<FLT>
         typename FArr  // fArr[m] yields Complex<FLT>
         >
void dirft1d1(const BIGINT nj,
              const XArr &x,
              const CArr &c,
              const int iflag,
              const BIGINT ms,
              FArr &fArr) {
  using Complex = std::decay_t<decltype(c[0])>;
  using FLT     = typename Complex::value_type;

  static_assert(std::is_floating_point<FLT>::value,
                "dirft1d1: Complex::value_type must be a floating point");
  static_assert(std::is_same<std::decay_t<decltype(x[0])>, FLT>::value,
                "dirft1d1: xArr element type must match Complex::value_type");

  BIGINT kmin = -(ms / 2);
  Complex I0{0, FLT(iflag)};

  for (BIGINT m = 0; m < ms; ++m) fArr[m] = Complex{0, 0};

  for (BIGINT j = 0; j < nj; ++j) {
    FLT xj        = x[j];
    Complex phase = exp(I0 * xj);
    Complex p     = pow(phase, static_cast<FLT>(kmin));
    Complex cj    = c[j];
    for (BIGINT m = 0; m < ms; ++m) {
      fArr[m] += cj * p;
      p *= phase;
    }
  }
}

// Direct computation of 1D type-2 nonuniform FFT. Interface same as finufft1d2
// 1D type‑2 NUFFT, direct:
//   c[j] = Σ_{m=0..ms-1} f[m] exp(i*iflag * k_m * x[j])
//   for -ms/2 <= k1 <= (ms-1)/2.
//   The output array is in increasing k1 ordering. If iflag>0 the + sign is
//   used, otherwise the - sign is used, in the exponential.
//  Uses winding trick.  Barnett 1/25/17
template<typename BIGINT,
         typename XArr, // xArr[j] yields real FLT
         typename CArr, // cArr[j] yields Complex<FLT>
         typename FArr  // fArr[m] yields Complex<FLT>
         >
void dirft1d2(const BIGINT nj,
              const XArr &x,
              CArr &c,
              const int iflag,
              const BIGINT ms,
              const FArr &f) {
  using Complex = std::decay_t<decltype(c[0])>;
  using FLT     = typename Complex::value_type;

  static_assert(std::is_floating_point<FLT>::value,
                "dirft1d2: Complex::value_type must be a floating point");
  static_assert(std::is_same<std::decay_t<decltype(x[0])>, FLT>::value,
                "dirft1d2: xArr element type must match Complex::value_type");
  static_assert(std::is_integral_v<BIGINT>, "dirft1d2: BIGINT must be an integral type");

  BIGINT kmin = -(ms / 2); // integer divide
  Complex I0{0, FLT(iflag)};

  for (BIGINT j = 0; j < nj; ++j) {
    FLT xj        = x[j];
    Complex phase = exp(I0 * xj);
    Complex p = pow(phase, static_cast<FLT>(kmin)); // starting phase for most neg freq
    Complex sum{0, 0};
    for (BIGINT m = 0; m < ms; ++m) {
      sum += f[m] * p;
      p *= phase;
    }
    c[j] = sum;
  }
}

// Direct computation of 1D type-3 nonuniform FFT. Interface same as finufft1d3
// 1D type‑3 NUFFT, direct:
//   f[k] = Σ_{j=0..nj-1} c[j] exp(i*iflag * s[k] * x[j])
//                    for k = 0, ..., nk-1
//  If iflag>0 the + sign is used, otherwise the - sign is used, in the
//  exponential. Uses C++ complex type. Simple brute force.  Barnett 1/25/17
// ------------------------------------------------------------
template<typename BIGINT,
         typename XArr, // xArr[j] yields real FLT
         typename CArr, // cArr[j] yields Complex<FLT>
         typename SArr, // sArr[k] yields real FLT
         typename FArr  // fArr[k] yields Complex<FLT>
         >
void dirft1d3(const BIGINT nj,
              const XArr &x,
              const CArr &c,
              const int iflag,
              const BIGINT nk,
              const SArr &sArr,
              FArr &f) {
  using Complex = std::decay_t<decltype(c[0])>;
  using FLT     = typename Complex::value_type;

  static_assert(std::is_floating_point_v<FLT>,
                "dirft1d3: Complex::value_type must be a floating point");
  static_assert(std::is_same<std::decay_t<decltype(x[0])>, FLT>::value,
                "dirft1d3: xArr element type must match Complex::value_type");
  static_assert(std::is_same<std::decay_t<decltype(sArr[0])>, FLT>::value,
                "dirft1d3: sArr element type must match Complex::value_type");

  const Complex I0{0, FLT(iflag)};

  for (BIGINT k = 0; k < nk; ++k) {
    Complex ss = I0 * sArr[k];
    Complex sum{0, 0};
    for (BIGINT j = 0; j < nj; ++j) sum += c[j] * exp(ss * x[j]);
    f[k] = sum;
  }
}
