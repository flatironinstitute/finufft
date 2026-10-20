/* Example of calling the cuFINUFFT library from C++, using thrust device
   vectors, with a math test. GPU version of examples/threadsafe1d1.cpp.
   Independent transforms run concurrently, one plan per std::thread, each on
   its own cudaStream_t. Barbone 11/12/26 (issue #183).
   To compile, see README in this directory. Usage: ./threadsafe1d1
*/
#include <cufinufft.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cuda_runtime.h>
#include <random>
#include <thread>
#include <thrust/device_vector.h>
#include <vector>
using namespace std;

constexpr double PI = 3.141592653589793238462643383279502884;

int main() {
  constexpr int M      = 2e5;  // number of nonuniform points
  constexpr int N      = 1e5;  // number of modes
  constexpr double tol = 1e-9; // desired accuracy
  double maxerr        = 0.0;  // worst rel err over threads

  const int nthreads   = max(1u, min(8u, thread::hardware_concurrency()));
  vector<double> errs(nthreads, HUGE_VAL); // per-thread rel err, HUGE_VAL if failed

  // Now have each thread do independent 1D type 1 on their own data and stream:
  const auto worker = [&](int tid) {
    // generate some random nonuniform points (x) and complex strengths (c)...
    // Note that these are local to the thread (if you have the *same* sets of
    // NU pts x for each thread, consider instead using one vectorized transform,
    // which would be faster).
    mt19937 rng(12345 + tid); // per-thread fixed seed
    uniform_real_distribution<double> upi(-PI, PI), u1(-1.0, 1.0);
    vector<double> x(M);
    vector<complex<double>> c(M), F(N); // F: output modes, local to the thread
    generate(x.begin(), x.end(), [&] { return upi(rng); });
    generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });

    thrust::device_vector<double> d_x          = x;          // copy x to the GPU
    thrust::device_vector<complex<double>> d_c = c, d_f = F; // also c, and room for F
    auto *dc = reinterpret_cast<cuDoubleComplex *>(d_c.data().get()); // same layout
    auto *df = reinterpret_cast<cuDoubleComplex *>(d_f.data().get());

    cudaStream_t stream; // one stream per thread
    if (cudaStreamCreate(&stream) != cudaSuccess) return void(errs[tid] = HUGE_VAL);
    cufinufft_opts opts;
    cufinufft_default_opts(&opts);
    opts.gpu_stream     = stream;
    const int64_t Ns[3] = {N, 1, 1};
    cufinufft_plan plan;
    int ier             = cufinufft_makeplan(1, 1, Ns, +1, 1, tol, &plan, &opts);
    if (ier) return void(errs[tid] = HUGE_VAL);
    ier = cufinufft_setpts(plan, M, d_x.data().get(), nullptr, nullptr, 0, nullptr,
                           nullptr, nullptr);
    if (ier) return void(errs[tid] = HUGE_VAL);
    ier = cufinufft_execute(plan, dc, df);
    if (!ier && cudaStreamSynchronize(stream) != cudaSuccess) ier = 1;
    cufinufft_destroy(plan);
    cudaStreamDestroy(stream);
    thrust::copy(d_f.begin(), d_f.end(), F.begin()); // copy F back to the CPU

    double err = HUGE_VAL;                           // default to fail
    if (!ier) {
      constexpr int k = 42519; // check the answer just for this mode, in [-N/2,N/2)
      const complex<double> I(0.0, 1.0);
      complex<double> Ftest(0.0, 0.0);
      for (int j = 0; j < M; ++j) Ftest += c[j] * exp(I * double(k) * x[j]);
      double Fmax = 0.0; // compute inf norm of F
      for (const auto &Fm : F) {
        const double a = abs(Fm);
        if (a > Fmax) Fmax = a;
      }
      err = abs(F[k + N / 2] - Ftest) / Fmax;
    }
    errs[tid] = err;
  };
  vector<thread> threads;
  for (int t = 0; t < nthreads; ++t) threads.emplace_back(worker, t);
  for (auto &th : threads) th.join();

  bool ok = true;
  for (const double e : errs) {
    ok &= e <= 10 * tol; // written so a NaN error fails
    maxerr = e > maxerr ? e : maxerr;
  }
  if (!ok) {
    fprintf(stderr, "FAILED: ier or rel err %.3g > %.3g\n", maxerr, 10 * tol);
    return 1;
  }
  printf("rel err %.3g\n", maxerr);
  return 0;
}
