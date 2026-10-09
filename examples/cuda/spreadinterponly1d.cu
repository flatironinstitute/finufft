/* GPU mirror of examples/spreadinterponly1d.cpp: spread/interp-only tasks
   via cufinufft with opts.gpu_spreadinterponly=1, with basic math tests.
   Barnett 1/8/25; GPU mirror 10/1/26.
   Usage: ./spreadinterponly1d
*/

#include <cufinufft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cuda_runtime.h>
#include <numeric>
#include <random>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>

constexpr double pi = 3.14159265358979323846;

int main() {
  constexpr int64_t M = 1e7; // number of nonuniform points
  constexpr int64_t N = 1e7; // size of regular grid

  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.gpu_spreadinterponly = 1;
  opts.upsampfac            = 2.0;  // pretend upsampling factor (really none)
  constexpr double tol      = 1e-9; // tolerance for (real) kernel shape design only

  std::vector<double> x(M);         // input
  std::vector<std::complex<double>> c(M), F(N); // c: input; F: output (spread to this
                                                // array)

  // device arrays free themselves, throwing on failure
  // first spread M=1 single unit-strength at the origin, only to get its total mass...
  x[0]                              = 0.0;
  c[0]                              = 1.0;
  thrust::device_vector<double> d_x = x;
  auto *h_c                         = reinterpret_cast<cuDoubleComplex *>(c.data());
  auto *h_F                         = reinterpret_cast<cuDoubleComplex *>(F.data());
  thrust::device_vector<cuDoubleComplex> d_c(h_c, h_c + M), d_F(N);
  constexpr int unused = 1;
  int ier = cufinufft1d1(1, thrust::raw_pointer_cast(d_x.data()),
                         thrust::raw_pointer_cast(d_c.data()), unused, tol, N,
                         thrust::raw_pointer_cast(d_F.data()), &opts); // warm-up
  if (ier > 0 || cudaDeviceSynchronize() != cudaSuccess) return ier ? ier : 1;
  thrust::copy(d_F.begin(), d_F.end(), h_F);
  const auto kersum = std::reduce(F.begin(), F.end()); // kernel mass

  // Now random nonuniform points (x) and complex strengths (c), fixed seed...
  std::mt19937 rng{12345};
  std::uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);
  std::generate(x.begin(), x.end(), [&] { return upi(rng); });
  std::generate(c.begin(), c.end(),
                [&] { return std::complex<double>{u1(rng), u1(rng)}; });
  d_x = x;
  thrust::copy(h_c, h_c + M, d_c.begin());

  opts.debug = 1;
  auto t0    = std::chrono::steady_clock::now(); // now spread with all M pts... (dir=1)
  ier        = cufinufft1d1(M, thrust::raw_pointer_cast(d_x.data()),
                            thrust::raw_pointer_cast(d_c.data()), unused, tol, N,
                            thrust::raw_pointer_cast(d_F.data()), &opts); // do it
  const cudaError_t cerr = cudaDeviceSynchronize();
  auto t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (ier > 0 || cerr != cudaSuccess) return ier ? ier : 1;
  thrust::copy(d_F.begin(), d_F.end(), h_F);
  const auto csum = std::reduce(c.begin(), c.end()); // input
  const auto mass = std::reduce(F.begin(), F.end()); // output
  // normalize by sum|c|, since sum c may nearly cancel:
  std::vector<double> terms(M);
  std::transform(c.begin(), c.end(), terms.begin(), [](auto cj) { return std::abs(cj); });
  const auto asum   = std::reduce(terms.begin(), terms.end());
  const auto relerr = std::abs(mass - kersum * csum) / (std::abs(kersum) * asum);
  printf(
      "1D spread-only, double-prec GPU, %.3g s (%.3g NU pt/sec), ier=%d, mass err %.3g\n",
      t, M / t, ier, relerr);

  std::fill(F.begin(), F.end(), std::complex<double>{1.0, 0.0}); // unit grid input
  thrust::copy(h_F, h_F + N, d_F.begin());
  opts.debug = 0;
  t0         = std::chrono::steady_clock::now(); // now interp to all M pts...  (dir=2)
  ier        = cufinufft1d2(M, thrust::raw_pointer_cast(d_x.data()),
                            thrust::raw_pointer_cast(d_c.data()), unused, tol, N,
                            thrust::raw_pointer_cast(d_F.data()), &opts); // do it
  const cudaError_t cerr2 = cudaDeviceSynchronize();
  t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (ier > 0 || cerr2 != cudaSuccess) return ier ? ier : 1;
  thrust::copy(d_c.begin(), d_c.end(), h_c);
  std::vector<double> errs(M); // max |cj - kersum|; each should be the kernel mass
  std::transform(c.begin(), c.end(), errs.begin(),
                 [&](auto cj) { return std::abs(cj - kersum); });
  double maxerr = 0;
  for (int64_t i = 0; i < M; ++i)
    if (errs[i] > maxerr || !std::isfinite(errs[i])) maxerr = errs[i]; // NaN lands too
  if (!std::isfinite(maxerr) || !(maxerr < 10 * tol * std::abs(kersum)) ||
      !(relerr < 10 * tol)) {
    fprintf(stderr, "FAILED: output non-finite or max err %.3g > %.3g\n",
            maxerr / std::abs(kersum), 10 * tol);
    return 1;
  }
  printf("1D interp-only, double-prec GPU, %.3g s (%.3g NU pt/sec), ier=%d, max err "
         "%.3g\n",
         t, M / t, ier, maxerr / std::abs(kersum));
  return 0;
}
