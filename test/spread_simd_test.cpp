#include "utils/norms.hpp"
#include <finufft/test_defs.h>
#include <limits>

using namespace std;
using namespace finufft::utils;

int main() {
  unsigned int se = 1;
  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.spreadinterponly = 1;
  int isign             = +1;
  double tol            = 1e-6;
  const FLT thresh      = 50 * numeric_limits<FLT>::epsilon();
  int fail              = 0;

  // 1D spread
  {
    BIGINT M = 32, N = 64;
    vector<FLT> x(M);
    vector<CPX> c(M);
    vector<CPX> F0(N), F1(N), F2(N);
    for (BIGINT j = 0; j < M; ++j) {
      x[j] = PI * randm11r(&se);
      c[j] = crandm11r(&se);
    }
    opts.spread_simd = 0;
    FINUFFT1D1(M, x.data(), c.data(), isign, tol, N, F0.data(), &opts);
    opts.spread_simd = 1;
    FINUFFT1D1(M, x.data(), c.data(), isign, tol, N, F1.data(), &opts);
    opts.spread_simd = 2;
    FINUFFT1D1(M, x.data(), c.data(), isign, tol, N, F2.data(), &opts);
    FLT err0 = relerrtwonorm(N, F0.data(), F2.data());
    FLT err1 = relerrtwonorm(N, F1.data(), F2.data());
    printf("spread 1D relerr0 %.3g relerr1 %.3g\n", err0, err1);
    if (err0 > thresh || err1 > thresh) fail = 1;
  }

  // 2D spread
  {
    BIGINT M = 32, N1 = 32, N2 = 16;
    vector<FLT> x(M), y(M);
    vector<CPX> c(M);
    vector<CPX> F0(N1 * N2), F1(N1 * N2), F2(N1 * N2);
    for (BIGINT j = 0; j < M; ++j) {
      x[j] = PI * randm11r(&se);
      y[j] = PI * randm11r(&se);
      c[j] = crandm11r(&se);
    }
    opts.spread_simd = 0;
    FINUFFT2D1(M, x.data(), y.data(), c.data(), isign, tol, N1, N2, F0.data(), &opts);
    opts.spread_simd = 1;
    FINUFFT2D1(M, x.data(), y.data(), c.data(), isign, tol, N1, N2, F1.data(), &opts);
    opts.spread_simd = 2;
    FINUFFT2D1(M, x.data(), y.data(), c.data(), isign, tol, N1, N2, F2.data(), &opts);
    FLT err0 = relerrtwonorm(N1 * N2, F0.data(), F2.data());
    FLT err1 = relerrtwonorm(N1 * N2, F1.data(), F2.data());
    printf("spread 2D relerr0 %.3g relerr1 %.3g\n", err0, err1);
    if (err0 > thresh || err1 > thresh) fail = 1;
  }

  // 3D spread
  {
    BIGINT M = 32, N1 = 16, N2 = 16, N3 = 16;
    vector<FLT> x(M), y(M), z(M);
    vector<CPX> c(M);
    vector<CPX> F0(N1 * N2 * N3), F1(N1 * N2 * N3), F2(N1 * N2 * N3);
    for (BIGINT j = 0; j < M; ++j) {
      x[j] = PI * randm11r(&se);
      y[j] = PI * randm11r(&se);
      z[j] = PI * randm11r(&se);
      c[j] = crandm11r(&se);
    }
    opts.spread_simd = 0;
    FINUFFT3D1(M, x.data(), y.data(), z.data(), c.data(), isign, tol, N1, N2, N3,
               F0.data(), &opts);
    opts.spread_simd = 1;
    FINUFFT3D1(M, x.data(), y.data(), z.data(), c.data(), isign, tol, N1, N2, N3,
               F1.data(), &opts);
    opts.spread_simd = 2;
    FINUFFT3D1(M, x.data(), y.data(), z.data(), c.data(), isign, tol, N1, N2, N3,
               F2.data(), &opts);
    FLT err0 = relerrtwonorm(N1 * N2 * N3, F0.data(), F2.data());
    FLT err1 = relerrtwonorm(N1 * N2 * N3, F1.data(), F2.data());
    printf("spread 3D relerr0 %.3g relerr1 %.3g\n", err0, err1);
    if (err0 > thresh || err1 > thresh) fail = 1;
  }

  // 1D interp
  {
    BIGINT M = 32, N = 64;
    vector<FLT> x(M);
    vector<CPX> c0(M), c1(M), c2(M);
    vector<CPX> F(N);
    for (BIGINT j = 0; j < M; ++j) x[j] = PI * randm11r(&se);
    for (BIGINT m = 0; m < N; ++m) F[m] = crandm11r(&se);
    opts.spread_simd = 0;
    FINUFFT1D2(M, x.data(), c0.data(), isign, tol, N, F.data(), &opts);
    opts.spread_simd = 1;
    FINUFFT1D2(M, x.data(), c1.data(), isign, tol, N, F.data(), &opts);
    opts.spread_simd = 2;
    FINUFFT1D2(M, x.data(), c2.data(), isign, tol, N, F.data(), &opts);
    FLT err0 = relerrtwonorm(M, c0.data(), c2.data());
    FLT err1 = relerrtwonorm(M, c1.data(), c2.data());
    printf("interp 1D relerr0 %.3g relerr1 %.3g\n", err0, err1);
    if (err0 > thresh || err1 > thresh) fail = 1;
  }

  // 2D interp
  {
    BIGINT M = 32, N1 = 32, N2 = 16;
    vector<FLT> x(M), y(M);
    vector<CPX> c0(M), c1(M), c2(M);
    vector<CPX> F(N1 * N2);
    for (BIGINT j = 0; j < M; ++j) {
      x[j] = PI * randm11r(&se);
      y[j] = PI * randm11r(&se);
    }
    for (BIGINT m = 0; m < N1 * N2; ++m) F[m] = crandm11r(&se);
    opts.spread_simd = 0;
    FINUFFT2D2(M, x.data(), y.data(), c0.data(), isign, tol, N1, N2, F.data(), &opts);
    opts.spread_simd = 1;
    FINUFFT2D2(M, x.data(), y.data(), c1.data(), isign, tol, N1, N2, F.data(), &opts);
    opts.spread_simd = 2;
    FINUFFT2D2(M, x.data(), y.data(), c2.data(), isign, tol, N1, N2, F.data(), &opts);
    FLT err0 = relerrtwonorm(M, c0.data(), c2.data());
    FLT err1 = relerrtwonorm(M, c1.data(), c2.data());
    printf("interp 2D relerr0 %.3g relerr1 %.3g\n", err0, err1);
    if (err0 > thresh || err1 > thresh) fail = 1;
  }

  // 3D interp
  {
    BIGINT M = 32, N1 = 16, N2 = 16, N3 = 16;
    vector<FLT> x(M), y(M), z(M);
    vector<CPX> c0(M), c1(M), c2(M);
    vector<CPX> F(N1 * N2 * N3);
    for (BIGINT j = 0; j < M; ++j) {
      x[j] = PI * randm11r(&se);
      y[j] = PI * randm11r(&se);
      z[j] = PI * randm11r(&se);
    }
    for (BIGINT m = 0; m < N1 * N2 * N3; ++m) F[m] = crandm11r(&se);
    opts.spread_simd = 0;
    FINUFFT3D2(M, x.data(), y.data(), z.data(), c0.data(), isign, tol, N1, N2, N3,
               F.data(), &opts);
    opts.spread_simd = 1;
    FINUFFT3D2(M, x.data(), y.data(), z.data(), c1.data(), isign, tol, N1, N2, N3,
               F.data(), &opts);
    opts.spread_simd = 2;
    FINUFFT3D2(M, x.data(), y.data(), z.data(), c2.data(), isign, tol, N1, N2, N3,
               F.data(), &opts);
    FLT err0 = relerrtwonorm(M, c0.data(), c2.data());
    FLT err1 = relerrtwonorm(M, c1.data(), c2.data());
    printf("interp 3D relerr0 %.3g relerr1 %.3g\n", err0, err1);
    if (err0 > thresh || err1 > thresh) fail = 1;
  }

  return fail;
}
