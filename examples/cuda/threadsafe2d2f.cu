/* Example of calling the cuFINUFFT library from C++, using thrust device
   vectors, with a zero-output test. GPU version of examples/threadsafe2d2f.cpp.
   Single-prec 2D type-2 transforms, one plan per std::thread, each on its own
   cudaStream_t. The shell of a multi-coil/slice MRI reconstruction app.
   Barbone 11/12/26. To compile, see README in this directory.
   Usage: ./threadsafe2d2f
*/
#include <cufinufft.h>

#include <algorithm>
#include <complex>
#include <cstdio>
#include <cuda_runtime.h>
#include <thread>
#include <thrust/device_vector.h>
#include <vector>
using namespace std;

int test_cufinufft(cufinufft_opts *opts)
// self-contained small test that one single-prec cuFINUFFT2D2 has no error/crash
{
  constexpr int64_t n_rows = 256, n_cols = 256;   // 2d image size
  constexpr int64_t n_read = 512, n_spokes = 128; // some k-space point params
  constexpr int64_t M = n_read * n_spokes;        // how many k-space pts; MRI-specific
  vector<float> x(M), y(M);                       // bunch of zero input data
  vector<complex<float>> img(n_rows * n_cols);    // coeffs
  vector<complex<float>> ksp(M);                  // output array (vals @ k-space pts)

  thrust::device_vector<float> d_x = x, d_y = y;  // copy x, y to the GPU
  thrust::device_vector<complex<float>> d_img = img, d_ksp = ksp;
  auto *dksp = reinterpret_cast<cuFloatComplex *>(d_ksp.data().get()); // same layout
  auto *dimg = reinterpret_cast<cuFloatComplex *>(d_img.data().get());

  cufinufftf_plan plan;
  const int64_t Ns[3] = {n_rows, n_cols, 1};
  int ier             = cufinufftf_makeplan(2, 2, Ns, -1, 1, 1e-3, &plan, opts);
  if (ier) return ier;
  ier = cufinufftf_setpts(plan, M, d_x.data().get(), d_y.data().get(), nullptr, 0,
                          nullptr, nullptr, nullptr);
  if (ier) return ier;
  ier               = cufinufftf_execute(plan, dksp, dimg);
  const auto stream = static_cast<cudaStream_t>(opts->gpu_stream); // typed as void*
  if (!ier && cudaStreamSynchronize(stream) != cudaSuccess) ier = 1;
  cufinufftf_destroy(plan);
  if (ier) return ier;
  thrust::copy(d_ksp.begin(), d_ksp.end(), ksp.begin()); // copy ksp back to the CPU
  for (const auto &kv : ksp)
    if (kv != complex<float>{}) return 1; // exact zero output; rejects NaN/Inf too
  return 0;
}

int main() {
  constexpr int n_slices = 50; // number of transforms, parallelize over slices
  const int nthreads     = max(1u, min(8u, thread::hardware_concurrency()));
  vector<int> status(nthreads, 0);

  // each thread works through the slices on its own cudaStream_t:
  const auto worker = [&](int tid) {
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) return void(status[tid] = 1);
    cufinufft_opts opts;
    cufinufft_default_opts(&opts);
    opts.gpu_stream = stream;
    for (int i = tid; i < n_slices; i += nthreads) // round-robin over threads
      if (test_cufinufft(&opts) != 0) status[tid] = 1;
    cudaStreamDestroy(stream);
  };
  vector<thread> threads;
  for (int t = 0; t < nthreads; ++t) threads.emplace_back(worker, t);
  for (auto &th : threads) th.join();

  int overallstatus = 0;
  for (const int s : status) overallstatus |= s;
  if (overallstatus) {
    fprintf(stderr, "FAILED: a slice returned an error or a nonzero output\n");
    return 1;
  }
  printf("ok\n");
  return 0;
}
