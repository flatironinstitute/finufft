/* GPU mirror of examples/threadsafe2d2f.cpp: single-prec 2D type-2 transforms, one
   CUFINUFFT plan per std::thread, each on its own
   cudaStream_t. MRI-style shell; no math check, like the CPU twin.
   Barnett 11/22/23; GPU mirror 10/1/26. Usage: ./threadsafe2d2f (50 lines) */

#include <cufinufft.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>

int test_cufinufft(cudaStream_t stream, int tid)
// self-contained small test that one single-prec CUFINUFFT2D2 has no error/crash
{
  constexpr int64_t n_rows = 256, n_cols = 256;   // 2d image size
  constexpr int64_t n_read = 512, n_spokes = 128; // some k-space point params
  constexpr int64_t M    = n_read * n_spokes;     // how many k-space pts
  constexpr int64_t Npix = n_rows * n_cols;
  const std::vector<float> x(M, 0.0f), y(M, 0.0f);
  const std::vector<std::complex<float>> img(Npix);

  // device arrays free themselves; a failed allocation or copy throws (exit != 0)
  const thrust::device_vector<float> d_x = x, d_y = y;
  const auto *h_img = reinterpret_cast<const cuFloatComplex *>(img.data());
  thrust::device_vector<cuFloatComplex> d_ksp(M), d_img(h_img, h_img + Npix);

  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.gpu_stream = stream;

  cufinufftf_plan plan;
  const int64_t Ns[3] = {n_rows, n_cols, 1};
  int ier             = cufinufftf_makeplan(2, 2, Ns, -1, 1, 1e-3, &plan, &opts);
  if (ier > 1) return 1;
  ier = cufinufftf_setpts(plan, M, thrust::raw_pointer_cast(d_x.data()),
                          thrust::raw_pointer_cast(d_y.data()), nullptr, 0, nullptr,
                          nullptr, nullptr);
  if (ier > 1) return 1;
  ier = cufinufftf_execute(plan, thrust::raw_pointer_cast(d_ksp.data()),
                           thrust::raw_pointer_cast(d_img.data()));
  if (ier <= 1 && cudaStreamSynchronize(stream) != cudaSuccess) ier = 2;
  cufinufftf_destroy(plan);

  // the output must be exactly 0: zero image at zero k-space points
  std::vector<std::complex<float>> ksp(M);
  thrust::copy(d_ksp.begin(), d_ksp.end(),
               reinterpret_cast<cuFloatComplex *>(ksp.data()));
  if (ier <= 1)
    for (const auto &v : ksp)
      if (!std::isfinite(std::abs(v)) || v != 0.0f) ier = 2;

  std::cout << "\ttest_cufinufft: exit code " << ier << ", thread " << tid << std::endl;
  return ier > 1;
}

int main() {
  constexpr int n_slices = 50; // number of transforms, parallelize over slices
  const int nthreads = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
  std::atomic<int> next{0};
  int overallstatus = 0;

  auto worker       = [&](int tid) {
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) return void(overallstatus = 1);
    for (int i = next++; i < n_slices; i = next++)
      if (test_cufinufft(stream, tid) != 0) overallstatus = 1;
    cudaStreamDestroy(stream);
  };

  std::vector<std::thread> threads;
  for (int t = 0; t < nthreads; ++t) threads.emplace_back(worker, t);
  for (auto &th : threads) th.join();

  if (overallstatus != 0)
    std::fprintf(stderr, "FAILED: F non-finite or rel err tolerance exceeded\n");
  return overallstatus;
}
