#include <cufinufft.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numeric>
#include <random>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>

#include <cuda_runtime.h>

constexpr float pi = 3.14159265358979323846f;

int main()
/* Example calling guru C++ interface to CUFINUFFT library on the GPU, single-prec,
   mirroring examples/guru1d1f.cpp.
   Usage: ./guru1d1f
*/
{
  constexpr int M     = 1e5;                   // number of nonuniform points
  constexpr int N     = 1e4;                   // number of modes
  constexpr float tol = 1e-3f;                 // desired accuracy
  constexpr std::complex<float> I(0.0f, 1.0f); // the imaginary unit

  constexpr int type = 1, dim = 1;             // 1d1
  const int64_t Ns[3]   = {N, 0, 0}; // guru describes mode array by vector [N1,N2..]
  constexpr int ntransf = 1;         // we want to do a single transform at a time
  cufinufft_opts opts;               // demo how to change options away from defaults...
  cufinufft_default_opts(&opts);
  opts.debug = 1;          // example options change (pass nullptr below for default opts)
  cufinufftf_plan plan;    // single-prec plan: note the "f"
  int ier = cufinufftf_makeplan(type, dim, Ns, +1, ntransf, tol, &plan, &opts);
  if (ier > 1) return ier; // no plan to use; going on would segfault

  std::mt19937 rng(12345); // fixed seed, so the run is deterministic
  std::uniform_real_distribution<float> upi(-pi, pi), u1(-1.0f, 1.0f);

  // generate some random nonuniform points in [-pi,pi]
  std::vector<float> x(M);
  std::generate(x.begin(), x.end(), [&] { return upi(rng); });
  std::vector<std::complex<float>> c(M), F(N);

  const thrust::device_vector<float> d_x = x;
  thrust::device_vector<cuFloatComplex> d_c(M, thrust::no_init), d_F(N, thrust::no_init);
  const auto *h_c = reinterpret_cast<const cuFloatComplex *>(c.data());
  ier = cufinufftf_setpts(plan, M, thrust::raw_pointer_cast(d_x.data()), nullptr, nullptr,
                          0, nullptr, nullptr, nullptr);
  if (ier > 1) return ier;

  std::generate(c.begin(), c.end(),
                [&] { return std::complex<float>{u1(rng), u1(rng)}; });
  thrust::copy(h_c, h_c + M, d_c.begin());
  ier = cufinufftf_execute(plan, thrust::raw_pointer_cast(d_c.data()),
                           thrust::raw_pointer_cast(d_F.data()));
  if (ier > 1) return ier;
  // for fun, do another with same NU pts (no re-sorting), but new strengths...
  std::generate(c.begin(), c.end(),
                [&] { return std::complex<float>{u1(rng), u1(rng)}; });
  thrust::copy(h_c, h_c + M, d_c.begin());
  ier = cufinufftf_execute(plan, thrust::raw_pointer_cast(d_c.data()),
                           thrust::raw_pointer_cast(d_F.data()));
  if (ier > 1 || cudaDeviceSynchronize() != cudaSuccess) return ier > 1 ? ier : 1;
  thrust::copy(d_F.begin(), d_F.end(), reinterpret_cast<cuFloatComplex *>(F.data()));
  cufinufftf_destroy(plan); // don't forget! done with transforms of this size
  constexpr int n  = 1251;  // check the answer just for this mode, must be in [-N/2,N/2)
  std::vector<std::complex<float>> terms(M);
  std::transform(c.begin(), c.end(), x.begin(), terms.begin(),
                 [&](auto cj, auto xj) { return cj * std::exp(I * float(n) * xj); });
  const auto Ftest = std::reduce(terms.begin(), terms.end());
  const auto Fmax = std::abs(*std::max_element(
      F.begin(), F.end(), [](auto a, auto b) { return std::abs(a) < std::abs(b); }));
  const auto err  = std::abs(F[n + N / 2] - Ftest) / Fmax; // n + N/2: index of freq mode
                                                           // n
  printf(
      "guru 1D type-1 single-prec NUFFT done (GPU). ier=%d, rel err in F[%d] is %.3g\n",
      ier, n, err);
  return !(err < 10 * tol);
}
