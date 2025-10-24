/*
  Simple example of the 1D type-1 transform using std::complex.
  To compile:
      nvcc -o getting_started getting_started.cpp -lcufinufft
*/

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufinufft.h>
#include <vector>

static constexpr double PI = 3.141592653589793238462643383279502884;

int main() {
  // Problem size: number of nonuniform points (M) and grid size (N).
  const int M = 100000, N = 10000;

  // Size of the grid as an array.
  int64_t modes[1] = {N};

  // Host pointers: frequencies (x), coefficients (c), and output (f).
  std::vector<float> x(M);
  std::vector<std::complex<float>> c(M);
  std::vector<std::complex<float>> f(N);

  // Device pointers.
  float *d_x;
  cuFloatComplex *d_c, *d_f;

  // Store cufinufft plan.
  cufinufftf_plan plan;

  // Manual calculation at a single point idx.
  int idx;
  std::complex<float> f0;

  // Fill with random numbers.
  std::srand(42);
  for (int j = 0; j < M; ++j) {
    x[j]     = 2 * PI * ((float)std::rand() / RAND_MAX - 1);
    float re = 2 * ((float)std::rand()) / RAND_MAX - 1;
    float im = 2 * ((float)std::rand()) / RAND_MAX - 1;
    c[j]     = std::complex<float>(re, im);
  }

  // Allocate the device arrays and copy x and c.
  cudaMalloc(&d_x, M * sizeof(float));
  cudaMalloc(&d_c, M * sizeof(cuFloatComplex));
  cudaMalloc(&d_f, N * sizeof(cuFloatComplex));

  std::vector<cuFloatComplex> c_dev(M);
  for (int j = 0; j < M; ++j) c_dev[j] = make_cuFloatComplex(c[j].real(), c[j].imag());

  cudaMemcpy(d_x, x.data(), M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c_dev.data(), M * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

  // Make the cufinufft plan for 1D type-1 transform.
  cufinufftf_makeplan(1, 1, modes, 1, 1, 1e-6, &plan, nullptr);

  // Set the frequencies of the nonuniform points.
  cufinufftf_setpts(plan, M, d_x, nullptr, nullptr, 0, nullptr, nullptr, nullptr);

  // Actually execute the plan on the given coefficients and store the result
  // in the d_f array.
  cufinufftf_execute(plan, d_c, d_f);

  // Copy the result back onto the host.
  cudaMemcpy(f.data(), d_f, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

  // Destroy the plan and free the device arrays after we're done.
  cufinufftf_destroy(plan);
  cudaFree(d_x);
  cudaFree(d_c);
  cudaFree(d_f);

  // Pick an index to check the result of the calculation.
  idx = 4 * N / 7;
  printf("f[%d] = %lf + %lfi\n", idx, std::real(f[idx]), std::imag(f[idx]));

  // Calculate the result manually using the formula for the type-1
  // transform.
  f0 = 0;

  std::complex<float> I(-0.0, 1.0);
  for (int j = 0; j < M; ++j) {
    f0 += c[j] * std::exp(I * x[j] * float(idx - N / 2));
  }

  printf("f0[%d] = %lf + %lfi\n", idx, std::real(f0), std::imag(f0));

  return 0;
}
