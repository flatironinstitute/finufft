#pragma once

#include <cmath>
#include <complex>
#include <type_traits>
#ifdef __CUDACC__
#include <thrust/complex.h>
#endif

// This is basically a port of dirft2d.f from CMCL package, except with
// the 1/nj prefactors for type-1 removed.

// Direct computation of 3D type-1 nonuniform FFT. Interface same as finufft3d1.
// 3D Type-1 NUFFT, direct:
//   f[k1,k2,k3] = Σ_j c[j] exp(i * iflag * (k1 x[j] + k2 y[j] + k3 z[j]))
//   for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2,
//       -mu/2 <= k3 <= (mu-1)/2,
//   The output array is in increasing k1 ordering (fast), then increasing
//   k2 ordering (medium), then increasing k3 (fast). If iflag>0 the + sign is
//   used, otherwise the - sign is used, in the exponential.
// Uses winding trick.  Barnett 2/1/17
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

// Direct computation of 3D type-2 nonuniform FFT. Interface same as finufft3d2
// 3D Type-2 NUFFT, direct:
//   c[j] = Σ_k1,k2,k3 f[k1,k2,k3] exp(i * iflag * (k1 x[j] + k2 y[j] + k3 z[j]))
//                 for j = 0,...,nj-1
// where sum is over -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2,
//           -mu/2 <= k3 <= (mu-1)/2.
// The input array is in increasing k1 ordering (fast), then increasing
// k2 ordering (medium), then increasing k3 (fast).
// If iflag>0 the + sign is used, otherwise the - sign is used, in the
// exponential.
// Uses winding trick.  Barnett 2/1/17
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
