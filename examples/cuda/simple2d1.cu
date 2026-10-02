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

constexpr double pi = 3.14159265358979323846;

int main() {
  /* GPU mirror of examples/simple2d1.cpp: simple 2D type-1 CUFINUFFT call from C++,
     double precision, with a math test. Usage: ./simple2d1 */
  constexpr int M      = 1e6;                 // number of nonuniform points
  constexpr int N      = 1e6;                 // approximate total number of modes (N1*N2)
  constexpr double tol = 1e-6;                // desired accuracy
  constexpr std::complex<double> I(0.0, 1.0); // the imaginary unit

  // random non-uniform points (x,y) in [-pi,pi] and strengths (c) with components in
  // [-1,1], from a fixed-seed generator:
  std::mt19937 rng{12345};
  std::uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);
  std::vector<double> x(M), y(M);
  std::vector<std::complex<double>> c(M);
  std::generate(x.begin(), x.end(), [&] { return upi(rng); });
  std::generate(y.begin(), y.end(), [&] { return upi(rng); });
  std::generate(c.begin(), c.end(),
                [&] { return std::complex<double>{u1(rng), u1(rng)}; });

  // choose numbers of output Fourier coefficients in each dimension
  const int N1 = std::round(2.0 * std::sqrt(N));
  const int N2 = std::round(N / N1);

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
  if (ier > 1) return ier; // no valid output to read
  if (cudaDeviceSynchronize() != cudaSuccess) return 1;
  thrust::copy(d_f.begin(), d_f.end(), reinterpret_cast<cuDoubleComplex *>(F.data()));

  const int k1 = std::round(0.45 * N1); // check the answer for mode frequency (k1,k2)
  const int k2 = std::round(-0.35 * N2);

  std::vector<std::complex<double>> terms(M);
  std::transform(x.begin(), x.end(), y.begin(), terms.begin(), [&](auto xj, auto yj) {
    return std::exp(I * (double(k1) * xj + double(k2) * yj));
  });
  std::transform(terms.begin(), terms.end(), c.begin(), terms.begin(),
                 [](auto tj, auto cj) { return tj * cj; });
  const auto Ftest = std::reduce(terms.begin(), terms.end());

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
