#include <cufinufft.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>

constexpr double PI = 3.141592653589793238462643383279502884;

int main() {
  /* GPU mirror of examples/simple2d1.cpp: simple 2D type-1 CUFINUFFT call from C++,
     double precision, with a math test. Usage: ./simple2d1 */
  constexpr int M      = 2e6;                 // number of nonuniform points
  constexpr double tol = 1e-6;                // desired accuracy
  constexpr std::complex<double> I(0.0, 1.0); // the imaginary unit

  // random non-uniform points (x,y) in [-PI,PI] and strengths (c) with components in
  // [-1,1], from a fixed-seed generator:
  std::mt19937 rng{12345};
  std::uniform_real_distribution<double> upi(-PI, PI), u1(-1.0, 1.0);
  std::vector<double> x(M), y(M);
  std::vector<std::complex<double>> c(M);
  std::generate(x.begin(), x.end(), [&] { return upi(rng); });
  std::generate(y.begin(), y.end(), [&] { return upi(rng); });
  std::generate(c.begin(), c.end(),
                [&] { return std::complex<double>{u1(rng), u1(rng)}; });

  // choose numbers of output Fourier coefficients in each dimension
  const int N1 = 1000, N2 = 2000;

  // output array for the Fourier modes
  std::vector<std::complex<double>> F(N1 * N2);

  // device arrays free themselves; a failed allocation or copy throws (exit != 0)
  const thrust::device_vector<double> d_x = x, d_y = y;
  const auto *h_c = reinterpret_cast<const cuDoubleComplex *>(c.data());
  const thrust::device_vector<cuDoubleComplex> d_c(h_c, h_c + M);
  thrust::device_vector<cuDoubleComplex> d_f(F.size());

  int ier = cufinufft2d1(M, thrust::raw_pointer_cast(d_x.data()),
                         thrust::raw_pointer_cast(d_y.data()),
                         thrust::raw_pointer_cast(d_c.data()), +1, tol, N1, N2,
                         thrust::raw_pointer_cast(d_f.data()), nullptr);
  if (ier > 0) return ier; // no valid output to read
  if (cudaDeviceSynchronize() != cudaSuccess) return 1;
  thrust::copy(d_f.begin(), d_f.end(), reinterpret_cast<cuDoubleComplex *>(F.data()));

  const int k1 = std::round(0.45 * N1); // check the answer for mode frequency (k1,k2)
  const int k2 = std::round(-0.35 * N2);

  std::complex<double> Ftest(0.0, 0.0);
  for (int j = 0; j < M; ++j)
    Ftest += c[j] * std::exp(I * (double(k1) * x[j] + double(k2) * y[j]));

  double Fmax      = 0.0;
  for (const auto &v : F) {
    const double a = std::abs(v);
    if (a > Fmax || !std::isfinite(a)) Fmax = a;
  }

  // index in output array for this frequency pair (k1,k2), then relative error
  const int indexOut = k1 + N1 / 2 + (k2 + N2 / 2) * N1;
  const auto err     = std::abs(F[indexOut] - Ftest) / Fmax;
  if (!std::isfinite(Fmax) || !(err < 10 * tol)) {
    fprintf(stderr, "FAILED: F non-finite or rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  std::cout << "2D type-1 NUFFT (GPU) done. ier=" << ier << ", err in F[" << indexOut
            << "] rel to max(F) is " << std::setprecision(2) << err << std::endl;
  return 0;
}
