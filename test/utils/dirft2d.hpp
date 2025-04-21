#pragma once

#include <cmath>
#include <complex>
#include <type_traits>
#ifdef __CUDACC__
#include <thrust/complex.h>
#endif

// ------------------------------------------------------------
// 2D Type-1 NUFFT, direct:
//   f[k1,k2] = Σ_{j=0..nj-1} c[j] exp(i * iflag * (k1 x[j] + k2 y[j]))
// ------------------------------------------------------------
template<typename BIGINT,
         typename XYArr, // x[j], y[j] → FLT
         typename CArr,  // c[j]       → Complex<FLT>
         typename FArr   // f[m]       → Complex<FLT>
         >
void dirft2d1(BIGINT nj,
              const XYArr &x,
              const XYArr &y,
              const CArr &c,
              int iflag,
              BIGINT ms,
              BIGINT mt,
              FArr &f) {
  using Complex = std::decay_t<decltype(c[0])>;
  using FLT     = typename Complex::value_type;

  static_assert(std::is_floating_point<FLT>::value,
                "Complex<FLT> must be floating point");
  static_assert(std::is_same<std::decay_t<decltype(x[0])>, FLT>::value,
                "x[j] must be FLT");
  static_assert(std::is_same<std::decay_t<decltype(y[0])>, FLT>::value,
                "y[j] must be FLT");

  const BIGINT k1min = -(ms / 2), k2min = -(mt / 2);
  const Complex I0{0, FLT(iflag)};
  const BIGINT N = ms * mt;

  for (BIGINT m = 0; m < N; ++m) f[m] = Complex{0, 0};

  for (BIGINT j = 0; j < nj; ++j) {
    Complex a1  = exp(I0 * x[j]);
    Complex a2  = exp(I0 * y[j]);
    Complex sp1 = pow(a1, FLT(k1min));
    Complex p2  = pow(a2, FLT(k2min));
    Complex cc  = c[j];
    BIGINT m    = 0;
    for (BIGINT m2 = 0; m2 < mt; ++m2) {
      Complex p1 = sp1;
      for (BIGINT m1 = 0; m1 < ms; ++m1) {
        f[m++] += cc * p1 * p2;
        p1 *= a1;
      }
      p2 *= a2;
    }
  }
}

// ------------------------------------------------------------
// 2D Type-2 NUFFT, direct:
//   c[j] = Σ_{k1,k2} f[k1,k2] exp(i * iflag * (k1 x[j] + k2 y[j]))
// ------------------------------------------------------------
template<typename BIGINT, typename XYArr, typename CArr, typename FArr>
void dirft2d2(BIGINT nj,
              const XYArr &x,
              const XYArr &y,
              CArr &c,
              int iflag,
              BIGINT ms,
              BIGINT mt,
              const FArr &f) {
  using Complex = std::decay_t<decltype(c[0])>;
  using FLT     = typename Complex::value_type;

  static_assert(std::is_floating_point<FLT>::value,
                "Complex<FLT> must be floating point");
  static_assert(std::is_same<std::decay_t<decltype(x[0])>, FLT>::value,
                "x[j] must be FLT");
  static_assert(std::is_same<std::decay_t<decltype(y[0])>, FLT>::value,
                "y[j] must be FLT");

  const BIGINT k1min = -(ms / 2), k2min = -(mt / 2);
  const Complex I0{0, FLT(iflag)};

  for (BIGINT j = 0; j < nj; ++j) {
    Complex a1  = exp(I0 * x[j]);
    Complex a2  = exp(I0 * y[j]);
    Complex sp1 = pow(a1, FLT(k1min));
    Complex p2  = pow(a2, FLT(k2min));
    Complex acc{0, 0};
    BIGINT m = 0;
    for (BIGINT m2 = 0; m2 < mt; ++m2) {
      Complex p1 = sp1;
      for (BIGINT m1 = 0; m1 < ms; ++m1) {
        acc += f[m++] * p1 * p2;
        p1 *= a1;
      }
      p2 *= a2;
    }
    c[j] = acc;
  }
}

// ------------------------------------------------------------
// 2D Type-3 NUFFT, direct:
//   f[k] = Σ_{j=0..nj-1} c[j] exp(i * iflag * (s[k] x[j] + t[k] y[j]))
// ------------------------------------------------------------
template<typename BIGINT, typename XYArr, typename CArr, typename STArr, typename FArr>
void dirft2d3(BIGINT nj,
              const XYArr &x,
              const XYArr &y,
              const CArr &c,
              int iflag,
              BIGINT nk,
              const STArr &s,
              const STArr &t,
              FArr &f) {
  using Complex = std::decay_t<decltype(c[0])>;
  using FLT     = typename Complex::value_type;

  static_assert(std::is_floating_point<FLT>::value,
                "Complex<FLT> must be floating point");
  static_assert(std::is_same<std::decay_t<decltype(x[0])>, FLT>::value,
                "x[j] must be FLT");
  static_assert(std::is_same<std::decay_t<decltype(s[0])>, FLT>::value,
                "s[k] must be FLT");

  const Complex I0{0, FLT(iflag)};

  for (BIGINT k = 0; k < nk; ++k) {
    Complex sum{0, 0};
    Complex ss = I0 * s[k];
    Complex tt = I0 * t[k];
    for (BIGINT j = 0; j < nj; ++j) sum += c[j] * exp(ss * x[j] + tt * y[j]);
    f[k] = sum;
  }
}
