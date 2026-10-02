/*

  Simple example of the 1D type-1 transform. To compile, run

       nvcc -o getting_started getting_started.cu -lcufinufft

  followed by

       ./getting_started

  with the necessary paths set if the library is not installed in the standard
  directories. If the library has been compiled in the standard way, this means

       export CPATH="${CPATH:+${CPATH}:}../../include"
       export LIBRARY_PATH="${LIBRARY_PATH:+${LIBRARY_PATH}:}../../build"
       export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}../../build"

 */

// docs-start: gs-headers
#include <algorithm>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufinufft.h>
#include <numeric>
#include <random>
#include <vector>

constexpr float pi = 3.14159265358979323846f;
// docs-end: gs-headers

int main() {
  // docs-start: gs-params
  // Problem size: number of nonuniform points (M) and grid size (N).
  const int M = 100000, N = 10000;

  // Size of the grid as an array.
  int64_t modes[1] = {N};

  // Host pointers: frequencies (x), coefficients (c), and output (f).
  float *x;
  std::complex<float> *c;
  std::complex<float> *f;
  // docs-end: gs-params

  // docs-start: gs-device
  // Device pointers.
  float *d_x;
  cuFloatComplex *d_c, *d_f;

  // Store cufinufft plan.
  cufinufftf_plan plan;
  // docs-end: gs-device

  // Manual calculation at a single point idx.
  int idx;
  std::complex<float> f0;

  // docs-start: gs-fill
  // Allocate the host arrays.
  x = (float *)malloc(M * sizeof(float));
  c = (std::complex<float> *)malloc(M * sizeof(std::complex<float>));
  f = (std::complex<float> *)malloc(N * sizeof(std::complex<float>));

  // Fill with random numbers. Frequencies must be in the interval [-pi, pi]
  // while strengths can be any value.
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> upi(-pi, pi), u1(-1.0f, 1.0f);

  std::generate(x, x + M, [&] { return upi(rng); });
  std::generate(c, c + M, [&] { return std::complex<float>{u1(rng), u1(rng)}; });
  // docs-end: gs-fill

  // docs-start: gs-transfer
  // Allocate the device arrays and copy the x and c arrays.
  cudaMalloc(&d_x, M * sizeof(float));
  cudaMalloc(&d_c, M * sizeof(cuFloatComplex));
  cudaMalloc(&d_f, N * sizeof(cuFloatComplex));

  cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, M * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
  // docs-end: gs-transfer

  // docs-start: gs-plan
  // Make the cufinufft plan for a 1D type-1 transform with six digits of
  // tolerance. Any ier above 1 is an error; 1 is a warning and the result is
  // still usable.
  int ier = cufinufftf_makeplan(1, 1, modes, 1, 1, 1e-6, &plan, NULL);
  if (ier > 1) return ier;

  // Set the frequencies of the nonuniform points.
  ier = cufinufftf_setpts(plan, M, d_x, NULL, NULL, 0, NULL, NULL, NULL);
  if (ier > 1) return ier;

  // Actually execute the plan on the given coefficients and store the result
  // in the d_f array.
  ier = cufinufftf_execute(plan, d_c, d_f);
  if (ier > 1) return ier;
  // docs-end: gs-plan

  // docs-start: gs-back
  // Copy the result back onto the host.
  cudaMemcpy(f, d_f, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

  // Destroy the plan and free the device arrays after we're done.
  cufinufftf_destroy(plan);

  cudaFree(d_x);
  cudaFree(d_c);
  cudaFree(d_f);
  // docs-end: gs-back

  // Pick an index to check the result of the calculation.
  idx = 4 * N / 7;

  printf("f[%d] = %lf + %lfi\n", idx, std::real(f[idx]), std::imag(f[idx]));

  // Calculate the result manually using the formula for the type-1
  // transform.
  std::vector<std::complex<float>> terms(M);
  std::transform(c, c + M, x, terms.begin(), [&](auto cj, auto xj) {
    return cj * std::exp(std::complex<float>(0, 1) * (xj * (idx - N / 2)));
  });
  f0 = std::reduce(terms.begin(), terms.end());

  printf("f0[%d] = %lf + %lfi\n", idx, std::real(f0), std::imag(f0));

  // Finally free the host arrays.
  free(x);
  free(c);
  free(f);

  return 0;
}
