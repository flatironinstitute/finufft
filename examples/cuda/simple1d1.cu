// Mirror of examples/simple1d1.cpp on the GPU (cufinufft, double precision).
#include <cufinufft.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cuda_runtime.h>
#include <numeric>
#include <random>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>

constexpr double PI = 3.141592653589793238462643383279502884;

int main()
/* Example of calling the CUFINUFFT library from C++, using STL
   double complex vectors on the host, with a math test.
   Usage: ./simple1d1
*/
{
  constexpr int M      = 2e6;                 // number of nonuniform points
  constexpr int N      = 1e6;                 // number of output modes
  constexpr double tol = 1e-9;                // desired accuracy
  constexpr std::complex<double> I(0.0, 1.0); // the imaginary unit

  // generate nonuniform points (x) and complex strengths (c)...
  std::vector<double> x(M);
  std::vector<std::complex<double>> c(M), F(N);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> upi(-PI, PI), u1(-1.0, 1.0);
  std::generate(x.begin(), x.end(), [&] { return upi(rng); });
  std::generate(c.begin(), c.end(),
                [&] { return std::complex<double>{u1(rng), u1(rng)}; });

  // device arrays free themselves; a failed allocation or copy throws (exit != 0)
  const thrust::device_vector<double> d_x = x;
  const auto *h_c = reinterpret_cast<const cuDoubleComplex *>(c.data());
  thrust::device_vector<cuDoubleComplex> d_c(h_c, h_c + M), d_f(N);

  // call the NUFFT (with iflag=+1) on the device arrays...
  int ier = cufinufft1d1(M, thrust::raw_pointer_cast(d_x.data()),
                         thrust::raw_pointer_cast(d_c.data()), +1, tol, N,
                         thrust::raw_pointer_cast(d_f.data()), nullptr);
  if (ier > 0) return ier; // no valid output to read
  if (cudaDeviceSynchronize() != cudaSuccess) return 1;
  thrust::copy(d_f.begin(), d_f.end(), reinterpret_cast<cuDoubleComplex *>(F.data()));

  constexpr int k = 142519; // check the answer just for this mode frequency...
  assert(k >= -(double)N / 2 && k < (double)N / 2);
  std::complex<double> Ftest(0.0, 0.0);
  for (int j = 0; j < M; ++j) Ftest += c[j] * std::exp(I * double(k) * x[j]);
  double Fmax      = 0.0;
  for (const auto &v : F) {
    const double a = std::abs(v);
    if (a > Fmax || !std::isfinite(a)) Fmax = a;
  }
  const auto err = std::abs(F[k + N / 2] - Ftest) / Fmax; // k + N/2: index of mode k
  if (!std::isfinite(Fmax) || !(err < 10 * tol)) {
    std::fprintf(stderr, "FAILED: F non-finite or rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  std::printf(
      "1D type-1 double-prec NUFFT (GPU) done. ier=%d, rel err in F[%d] is %.3g\n", ier,
      k, err);
  return 0;
}
