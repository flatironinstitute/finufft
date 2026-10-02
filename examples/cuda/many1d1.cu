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

constexpr double pi = 3.14159265358979323846;

int main()
/* GPU mirror of examples/many1d1.cpp: vectorized CUFINUFFT from C++, STL double
   complex vectors on the host, with a math test. Usage: ./many1d1 */
{
  constexpr int ntrans = 3;    // how many stacked transforms to do
  constexpr int M      = 1e6;  // nonuniform points (same for all transforms)
  constexpr int N      = 1e6;  // number of modes (same for all transforms)
  constexpr double tol = 1e-9; // desired accuracy
  constexpr std::complex<double> I{0.0, 1.0}; // the imaginary unit

  // random nonuniform points x in [-pi,pi] and strengths c, from a fixed seed...
  std::mt19937 rng{12345};
  std::uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);
  std::vector<double> x(M);
  std::vector<std::complex<double>> c(M * ntrans), F(N * ntrans); // F: output Fourier
                                                                  // modes
  std::generate(x.begin(), x.end(), [&] { return upi(rng); });
  std::generate(c.begin(), c.end(),
                [&] { return std::complex<double>{u1(rng), u1(rng)}; });

  // device arrays free themselves; a failed allocation or copy throws (exit != 0)
  const thrust::device_vector<double> d_x = x;
  const auto *h_c = reinterpret_cast<const cuDoubleComplex *>(c.data());
  const thrust::device_vector<cuDoubleComplex> d_c(h_c, h_c + c.size());
  thrust::device_vector<cuDoubleComplex> d_f(F.size(), thrust::no_init);

  int ier = cufinufft1d1many(ntrans, M, thrust::raw_pointer_cast(d_x.data()),
                             thrust::raw_pointer_cast(d_c.data()), +1, tol, N,
                             thrust::raw_pointer_cast(d_f.data()), nullptr);
  if (ier > 1) return ier; // no valid output to read
  if (cudaDeviceSynchronize() != cudaSuccess) return 1;
  thrust::copy(d_f.begin(), d_f.end(), reinterpret_cast<cuDoubleComplex *>(F.data()));

  constexpr int k     = 142519;               // check the answer just for this mode...
  constexpr int trans = ntrans - 1;           // ...in this transform
  assert(k >= -(double)N / 2 && k < (double)N / 2);
  std::vector<std::complex<double>> terms(M); // naive calc, c from transform # trans
  std::transform(x.begin(), x.end(), c.begin() + M * trans, terms.begin(),
                 [&](auto xj, auto cj) { return cj * std::exp(I * double(k) * xj); });
  const auto Ftest = std::reduce(terms.begin(), terms.end());
  const auto Ft    = F.begin() + N * trans;
  const auto Fmax  = std::abs(*std::max_element(
      Ft, Ft + N, [](auto a, auto b) { return std::abs(a) < std::abs(b); }));
  const auto err = std::abs(F[k + N / 2 + N * trans] - Ftest) / Fmax; // output index of k
  printf("1D type-1 double-prec NUFFT (GPU) done. ier=%d, rel err in F[%d] is %.3g\n",
         ier, k, err);
  return !(err < 10 * tol);
}
