/* GPU mirror of examples/threadsafe1d1.cpp: independent CUFINUFFT transforms run
   concurrently, one plan per std::thread, each on its
   own cudaStream_t. Barnett 4/19/21; GPU mirror 10/1/26. Usage: ./threadsafe1d1 */

#include <cufinufft.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cuda_runtime.h>
#include <numeric>
#include <random>
#include <thread>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>

constexpr double pi = 3.14159265358979323846;

int main() {
  constexpr int M      = 1e5;                 // number of nonuniform points
  constexpr int N      = 1e5;                 // number of modes
  constexpr double tol = 1e-9;                // desired accuracy
  constexpr std::complex<double> I{0.0, 1.0}; // the imaginary unit

  const int nthreads = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
  int overallstatus  = 0;

  // Now have each thread do independent 1D type 1 on their own data:
  const auto worker  = [&](int tid) {
    // random nonuniform points (x) and complex strengths (c), local to the thread
    std::mt19937 rng{unsigned(12345 + tid)}; // per-thread fixed seed
    std::uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);
    std::vector<double> x(M);
    std::vector<std::complex<double>> c(M), F(N); // F: output modes, local to the thread
    std::generate(x.begin(), x.end(), [&] { return upi(rng); });
    std::generate(c.begin(), c.end(),
                  [&] { return std::complex<double>{u1(rng), u1(rng)}; });

    // device arrays free themselves; a failed allocation or copy throws (exit != 0)
    const thrust::device_vector<double> d_x = x;
    const auto *h_c = reinterpret_cast<const cuDoubleComplex *>(c.data());
    thrust::device_vector<cuDoubleComplex> d_c(h_c, h_c + M), d_F(N);

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) return void(overallstatus = 1);
    cufinufft_opts opts;
    cufinufft_default_opts(&opts);
    opts.gpu_stream = stream;

    cufinufft_plan plan;
    const int64_t Ns[3] = {N, 1, 1};
    int ier             = cufinufft_makeplan(1, 1, Ns, +1, 1, tol, &plan, &opts);
    if (ier > 1) return void(overallstatus = 1);
    ier = cufinufft_setpts(plan, M, thrust::raw_pointer_cast(d_x.data()), nullptr,
                           nullptr, 0, nullptr, nullptr, nullptr);
    if (ier > 1) return void(overallstatus = 1);
    ier = cufinufft_execute(plan, thrust::raw_pointer_cast(d_c.data()),
                            thrust::raw_pointer_cast(d_F.data()));
    if (ier > 1 || cudaStreamSynchronize(stream) != cudaSuccess)
      return void(overallstatus = 1);
    cufinufft_destroy(plan);
    cudaStreamDestroy(stream);
    thrust::copy(d_F.begin(), d_F.end(), reinterpret_cast<cuDoubleComplex *>(F.data()));
    constexpr int k = 42519; // check the answer just for this mode frequency...
    std::vector<std::complex<double>> terms(M);
    std::transform(c.begin(), c.end(), x.begin(), terms.begin(),
                   [&](auto cj, auto xj) { return cj * std::exp(I * double(k) * xj); });
    const auto Ftest = std::reduce(terms.begin(), terms.end());
    double Fmax      = 0.0;
    for (const auto &v : F) {
      const double a = std::abs(v);
      if (a > Fmax || !std::isfinite(a)) Fmax = a;
    }
    const auto err = std::abs(F[k + N / 2] - Ftest) / Fmax; // k + N/2: index of freq
                                                            // mode k
    if (!std::isfinite(Fmax) || !(err < 10 * tol)) overallstatus = 1;
    printf("[thread %2d] 1D t-1 dbl-prec GPU NUFFT done. ier=%d, rel err in F[%d]: "
           "%.3g\n",
           tid, ier, k, err);
  };
  std::vector<std::thread> threads;
  for (int t = 0; t < nthreads; ++t) threads.emplace_back(worker, t);
  for (auto &th : threads) th.join();
  if (overallstatus != 0) {
    printf("FAILED: F non-finite or rel err > %.3g\n", 10 * tol);
    return 1;
  }
  return overallstatus;
}
