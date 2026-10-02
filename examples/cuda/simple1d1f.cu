// Mirror of examples/simple1d1f.cpp on the GPU (cufinufft, single precision).
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

constexpr float pi = 3.14159265358979323846f;

int main()
/* Example of calling the CUFINUFFT library from C++, using STL
   single complex vectors on the host, with a math test.
   Usage: ./simple1d1f
*/
{
  constexpr int M     = 1e5;                   // number of nonuniform points
  constexpr int N     = 1e4;                   // number of modes
  constexpr float acc = 1e-3;                  // desired accuracy
  constexpr std::complex<float> I(0.0f, 1.0f); // the imaginary unit

  // generate nonuniform points (x) and complex strengths (c)...
  std::vector<float> x(M);
  std::vector<std::complex<float>> c(M), F(N);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> upi(-pi, pi), u1(-1.0f, 1.0f);
  std::generate(x.begin(), x.end(), [&] { return upi(rng); });
  std::generate(c.begin(), c.end(),
                [&] { return std::complex<float>{u1(rng), u1(rng)}; });

  // device arrays free themselves; a failed allocation or copy throws (exit != 0)
  const thrust::device_vector<float> d_x = x;
  const auto *h_c = reinterpret_cast<const cuFloatComplex *>(c.data());
  thrust::device_vector<cuFloatComplex> d_c(h_c, h_c + M), d_f(N, thrust::no_init);

  // call the NUFFT (with iflag=+1) on the device arrays...
  int ier = cufinufftf1d1(M, thrust::raw_pointer_cast(d_x.data()),
                          thrust::raw_pointer_cast(d_c.data()), +1, acc, N,
                          thrust::raw_pointer_cast(d_f.data()), nullptr); // note "f"
  if (ier > 1) return ier; // no valid output to read
  if (cudaDeviceSynchronize() != cudaSuccess) return 1;
  thrust::copy(d_f.begin(), d_f.end(), reinterpret_cast<cuFloatComplex *>(F.data()));

  constexpr int k = 1425; // check the answer just for this mode...
  assert(k >= -(double)N / 2 && k < (double)N / 2);
  std::vector<std::complex<float>> terms(M);
  std::transform(c.begin(), c.end(), x.begin(), terms.begin(),
                 [&](auto cj, auto xj) { return cj * std::exp(I * float(k) * xj); });
  const auto Ftest = std::reduce(terms.begin(), terms.end());
  const auto Fmax = std::abs(*std::max_element(
      F.begin(), F.end(), [](auto a, auto b) { return std::abs(a) < std::abs(b); }));
  const auto err  = std::abs(F[k + N / 2] - Ftest) / Fmax; // k + N/2: index of mode k
  std::printf(
      "1D type-1 single-prec NUFFT (GPU) done. ier=%d, rel err in F[%d] is %.3g\n", ier,
      k, err);
  return !(err < 10 * acc);
}
