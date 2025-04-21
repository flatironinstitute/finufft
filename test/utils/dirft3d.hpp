#pragma once

#include <cmath>
#include <complex>
#include <type_traits>
#ifdef __CUDACC__
#include <thrust/complex.h>
#endif

// ------------------------------------------------------------
// 3D Type-1 NUFFT, direct:
//   f[k1,k2,k3] = Σ_j c[j] exp(i * iflag * (k1 x[j] + k2 y[j] + k3 z[j]))
// ------------------------------------------------------------
template<typename BIGINT,
         typename XYZArr, // x[j], y[j], z[j] → FLT
         typename CArr,   // c[j]             → Complex<FLT>
         typename FArr>   // f[m]             → Complex<FLT>
void dirft3d1(BIGINT nj,
              const XYZArr &x,
              const XYZArr &y,
              const XYZArr &z,
              const CArr &c,
              int iflag,
              BIGINT ms,
              BIGINT mt,
              BIGINT mu,
              FArr &f) {
  using Complex = std::decay_t<decltype(c[0])>;
  using FLT     = typename Complex::value_type;

  static_assert(std::is_floating_point<FLT>::value,
                "Complex<FLT> must be floating point");
  static_assert(std::is_same<std::decay_t<decltype(x[0])>, FLT>::value,
                "x[j] must be FLT");

  const BIGINT k1min = -(ms / 2), k2min = -(mt / 2), k3min = -(mu / 2);
  const Complex I0{0, FLT(iflag)};
  const BIGINT N = ms * mt * mu;

  for (BIGINT m = 0; m < N; ++m) f[m] = Complex{0, 0};

  for (BIGINT j = 0; j < nj; ++j) {
    Complex a1  = exp(I0 * x[j]);
    Complex a2  = exp(I0 * y[j]);
    Complex a3  = exp(I0 * z[j]);
    Complex sp1 = pow(a1, FLT(k1min));
    Complex sp2 = pow(a2, FLT(k2min));
    Complex p3  = pow(a3, FLT(k3min));
    Complex cc  = c[j];

    BIGINT m = 0;
    for (BIGINT m3 = 0; m3 < mu; ++m3) {
      Complex p2 = sp2;
      for (BIGINT m2 = 0; m2 < mt; ++m2) {
        Complex p1 = sp1;
        for (BIGINT m1 = 0; m1 < ms; ++m1) {
          f[m++] += cc * p1 * p2 * p3;
          p1 *= a1;
        }
        p2 *= a2;
      }
      p3 *= a3;
    }
  }
}

// ------------------------------------------------------------
// 3D Type-2 NUFFT, direct:
//   c[j] = Σ_k1,k2,k3 f[k1,k2,k3] exp(i * iflag * (k1 x[j] + k2 y[j] + k3 z[j]))
// ------------------------------------------------------------
template<typename BIGINT, typename XYZArr, typename CArr, typename FArr>
void dirft3d2(BIGINT nj,
              const XYZArr &x,
              const XYZArr &y,
              const XYZArr &z,
              CArr &c,
              int iflag,
              BIGINT ms,
              BIGINT mt,
              BIGINT mu,
              const FArr &f) {
  using Complex = std::decay_t<decltype(c[0])>;
  using FLT     = typename Complex::value_type;

  static_assert(std::is_floating_point<FLT>::value,
                "Complex<FLT> must be floating point");
  static_assert(std::is_same<std::decay_t<decltype(x[0])>, FLT>::value,
                "x[j] must be FLT");

  const BIGINT k1min = -(ms / 2), k2min = -(mt / 2), k3min = -(mu / 2);
  const Complex I0{0, FLT(iflag)};

  for (BIGINT j = 0; j < nj; ++j) {
    Complex a1  = exp(I0 * x[j]);
    Complex a2  = exp(I0 * y[j]);
    Complex a3  = exp(I0 * z[j]);
    Complex sp1 = pow(a1, FLT(k1min));
    Complex sp2 = pow(a2, FLT(k2min));
    Complex p3  = pow(a3, FLT(k3min));
    Complex acc{0, 0};

    BIGINT m = 0;
    for (BIGINT m3 = 0; m3 < mu; ++m3) {
      Complex p2 = sp2;
      for (BIGINT m2 = 0; m2 < mt; ++m2) {
        Complex p1 = sp1;
        for (BIGINT m1 = 0; m1 < ms; ++m1) {
          acc += f[m++] * p1 * p2 * p3;
          p1 *= a1;
        }
        p2 *= a2;
      }
      p3 *= a3;
    }
    c[j] = acc;
  }
}

// ------------------------------------------------------------
// 3D Type-3 NUFFT, direct:
//   f[k] = Σ_j c[j] exp(i * iflag * (s[k] x[j] + t[k] y[j] + u[k] z[j]))
// ------------------------------------------------------------
template<typename BIGINT, typename XYZArr, typename CArr, typename STUArr, typename FArr>
void dirft3d3(BIGINT nj,
              const XYZArr &x,
              const XYZArr &y,
              const XYZArr &z,
              const CArr &c,
              int iflag,
              BIGINT nk,
              const STUArr &s,
              const STUArr &t,
              const STUArr &u,
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
    Complex ss = I0 * s[k];
    Complex tt = I0 * t[k];
    Complex uu = I0 * u[k];
    Complex acc{0, 0};
    for (BIGINT j = 0; j < nj; ++j) acc += c[j] * exp(ss * x[j] + tt * y[j] + uu * z[j]);
    f[k] = acc;
  }
}
