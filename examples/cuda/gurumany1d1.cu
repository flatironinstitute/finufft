/* Demonstrate guru CUFINUFFT interface performing a stack of 1d type 1
   transforms in a single execute call, on the GPU. Mirrors
   examples/gurumany1d1.cpp.
   Usage: ./gurumany1d1           (exit code 0 indicates success)
*/

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

int main() {
  constexpr int M      = 2e5;  // number of nonuniform points
  constexpr int N      = 1e5;  // number of modes
  constexpr double tol = 1e-9; // desired accuracy
  constexpr int ntrans = 100;  // request a bunch of transforms in the execute
  constexpr int isign  = +1;   // sign of i in the transform math definition
  constexpr std::complex<double> I(0.0, 1.0); // the imaginary unit

  constexpr int type = 1, dim = 1;            // 1d1
  const int64_t Ns[3] = {N, 0, 0}; // guru describes mode array by vector [N1,N2..]
  cufinufft_plan plan;
  // nullptr here means use default opts...
  int ier = cufinufft_makeplan(type, dim, Ns, isign, ntrans, tol, &plan, nullptr);
  if (ier > 1) return ier; // no plan to use; going on would segfault

  std::mt19937 rng(12345); // fixed seed, so the run is deterministic
  std::uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);

  // generate random nonuniform points in [-pi,pi] and pass to CUFINUFFT
  std::vector<double> x(M);
  std::generate(x.begin(), x.end(), [&] { return upi(rng); });
  const thrust::device_vector<double> d_x = x;
  ier = cufinufft_setpts(plan, M, thrust::raw_pointer_cast(d_x.data()), nullptr, nullptr,
                         0, nullptr, nullptr, nullptr);
  if (ier > 1) return ier;

  // generate ntrans complex strength vectors each of length M (the slow bit!)
  std::vector<std::complex<double>> c(M * ntrans), F(N * ntrans); // plain contiguous
                                                                  // storage
  std::generate(c.begin(), c.end(),
                [&] { return std::complex<double>{u1(rng), u1(rng)}; });

  // device arrays free themselves; a failed allocation or copy throws (exit != 0)
  const auto *h_c = reinterpret_cast<const cuDoubleComplex *>(c.data());
  thrust::device_vector<cuDoubleComplex> d_c(h_c, h_c + c.size());
  thrust::device_vector<cuDoubleComplex> d_F(F.size());
  printf("guru many 1D type-1 double-prec (GPU), tol=%.3g, executing %d transforms "
         "(vectorized), each size %d NU pts to %d modes...\n",
         tol, ntrans, M, N);
  ier = cufinufft_execute(plan, thrust::raw_pointer_cast(d_c.data()),
                          thrust::raw_pointer_cast(d_F.data()));
  if (ier > 1 || cudaDeviceSynchronize() != cudaSuccess) return ier > 1 ? ier : 1;
  thrust::copy(d_F.begin(), d_F.end(), reinterpret_cast<cuDoubleComplex *>(F.data()));
  cufinufft_destroy(plan); // don't forget! we're done with transforms of this size
  // rest is math checking and reporting...
  constexpr int k     = 42519;      // check the answer just for this mode...
  constexpr int trans = ntrans - 1; // ...in this transform
  assert(k >= -(double)N / 2 && k < (double)N / 2); // ensure meaningful test
  assert(trans >= 0 && trans < ntrans);
  std::vector<std::complex<double>> terms(M);
  std::transform(
      x.begin(), x.end(), c.begin() + M * trans, terms.begin(),
      [&](auto xj, auto cj) { return cj * std::exp(I * double(k) * xj); }); // c offset
  const auto Ftest = std::reduce(terms.begin(), terms.end());
  const auto Ft    = F.begin() + N * trans; // output of the selected transform
  double Fmax      = 0.0;
  for (const auto &v : F) {
    const double a = std::abs(v);
    if (a > Fmax || !std::isfinite(a)) Fmax = a;
  }
  const auto err = std::abs(F[k + N / 2 + N * trans] - Ftest) / Fmax; // mode k index
  if (!std::isfinite(Fmax) || !(err < 10 * tol)) {
    fprintf(stderr, "FAILED: F non-finite or rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  printf("\tdone: ier=%d; for transform %d, rel err in F[%d] is %.3g\n", ier, trans, k,
         err);
  return 0;
}
