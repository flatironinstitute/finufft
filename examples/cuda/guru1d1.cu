#include <cufinufft.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numeric>
#include <random>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>

#include <cuda_runtime.h>

constexpr double pi = 3.14159265358979323846;

int main()
/* Example calling guru C++ interface to CUFINUFFT library on the GPU, mirroring
   examples/guru1d1.cpp: STL vectors on the host, thrust::device_vector for the device.
   Usage: ./guru1d1
*/
{
  constexpr int M      = 1e6;                 // number of nonuniform points
  constexpr int N      = 1e6;                 // number of modes
  constexpr double tol = 1e-9;                // desired accuracy
  constexpr std::complex<double> I(0.0, 1.0); // the imaginary unit

  constexpr int type = 1, dim = 1;            // 1d1
  const int64_t Ns[3]   = {N, 0, 0}; // guru describes mode array by vector [N1,N2..]
  constexpr int ntransf = 1;         // we want to do a single transform at a time
  cufinufft_plan plan;
  int ier = cufinufft_makeplan(type, dim, Ns, +1, ntransf, tol, &plan, nullptr);
  if (ier > 1) return ier; // no plan to use; going on would segfault (1 = warning)

  std::mt19937 rng(12345); // fixed seed, so the run is deterministic
  std::uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);

  // generate some random nonuniform points in [-pi,pi]
  std::vector<double> x(M);
  std::generate(x.begin(), x.end(), [&] { return upi(rng); });
  std::vector<std::complex<double>> c(M), F(N);

  const thrust::device_vector<double> d_x = x;
  thrust::device_vector<cuDoubleComplex> d_c(M), d_F(N);
  const auto *h_c = reinterpret_cast<const cuDoubleComplex *>(c.data());
  ier = cufinufft_setpts(plan, M, thrust::raw_pointer_cast(d_x.data()), nullptr, nullptr,
                         0, nullptr, nullptr, nullptr);
  if (ier > 1) return ier;

  std::generate(c.begin(), c.end(),
                [&] { return std::complex<double>{u1(rng), u1(rng)}; });
  thrust::copy(h_c, h_c + M, d_c.begin());
  ier = cufinufft_execute(plan, thrust::raw_pointer_cast(d_c.data()),
                          thrust::raw_pointer_cast(d_F.data()));
  if (ier > 1) return ier;
  // for fun, do another with same NU pts (no re-sorting), but new strengths...
  std::generate(c.begin(), c.end(),
                [&] { return std::complex<double>{u1(rng), u1(rng)}; });
  thrust::copy(h_c, h_c + M, d_c.begin());
  ier = cufinufft_execute(plan, thrust::raw_pointer_cast(d_c.data()),
                          thrust::raw_pointer_cast(d_F.data()));
  if (ier > 1 || cudaDeviceSynchronize() != cudaSuccess) return ier > 1 ? ier : 1;
  thrust::copy(d_F.begin(), d_F.end(), reinterpret_cast<cuDoubleComplex *>(F.data()));
  cufinufft_destroy(plan);  // don't forget! done with transforms of this size
  constexpr int n = 142519; // check the answer just for this mode
  assert(n >= -(double)N / 2 && n < (double)N / 2); // ensure meaningful test
  std::vector<std::complex<double>> terms(M);
  std::transform(c.begin(), c.end(), x.begin(), terms.begin(),
                 [&](auto cj, auto xj) { return cj * std::exp(I * double(n) * xj); });
  const auto Ftest = std::reduce(terms.begin(), terms.end());
  double Fmax      = 0.0;
  for (const auto &v : F) {
    const double a = std::abs(v);
    if (a > Fmax || !std::isfinite(a)) Fmax = a;
  }
  const auto err = std::abs(F[n + N / 2] - Ftest) / Fmax; // n + N/2: index of freq mode n
  if (!std::isfinite(Fmax) || !(err < 10 * tol)) {
    fprintf(stderr, "FAILED: F non-finite or rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  printf(
      "guru 1D type-1 double-prec NUFFT done (GPU). ier=%d, rel err in F[%d] is %.3g\n",
      ier, n, err);
  return 0;
}
