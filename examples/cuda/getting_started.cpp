/*

  Simple example of the 1D type-1 transform. To compile, run

       nvcc -o getting_started getting_started.cpp -lcufinufft

  followed by

       ./getting_started

  with the necessary paths set if the library is not installed in the standard
  directories. If the library has been compiled in the standard way, this means

       export CPATH="${CPATH:+${CPATH}:}../../include"
       export LIBRARY_PATH="${LIBRARY_PATH:+${LIBRARY_PATH}:}../../build"
       export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}../../build"

 */

// docs-start: getting_started-includes
#include <complex.h>
#include <cuComplex.h>
#include <cufinufft.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
// docs-end: getting_started-includes

constexpr double PI = 3.141592653589793238462643383279502884;

#include <cuda_runtime.h>

int main() {
  // docs-start: getting_started-main
  const int M = 100000, N = 10000; // number of NU points, grid size

  int64_t modes[1] = {N};          // grid size as an array

  float *x;                        // NU points
  float _Complex *c;               // strengths
  float _Complex *f;               // output

  float *d_x;                      // device (GPU) pointers
  cuFloatComplex *d_c, *d_f;

  cufinufftf_plan plan;

  int idx; // for the check at one index
  float _Complex f0;

  x = (float *)malloc(M * sizeof(float));
  c = (float _Complex *)malloc(M * sizeof(float _Complex));
  f = (float _Complex *)malloc(N * sizeof(float _Complex));

  srand(0);

  for (int j = 0; j < M; ++j) { // x is 2pi-periodic, c is any
    x[j] = 2 * PI * ((float)rand() / RAND_MAX - 1);
    c[j] = (2 * (float)rand() / RAND_MAX - 1) + I * (2 * (float)rand() / RAND_MAX - 1);
  }

  cudaMalloc(&d_x, M * sizeof(float));
  cudaMalloc(&d_c, M * sizeof(float _Complex));
  cudaMalloc(&d_f, N * sizeof(float _Complex));

  cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, M * sizeof(float _Complex), cudaMemcpyHostToDevice);

  // type 1, 1D, +, 1 transf, tol
  int ier = cufinufftf_makeplan(1, 1, modes, 1, 1, 1e-6, &plan, NULL);
  if (ier > 0) return ier;

  ier = cufinufftf_setpts(plan, M, d_x, NULL, NULL, 0, NULL, NULL, NULL);
  if (ier > 0) return ier;

  ier = cufinufftf_execute(plan, d_c, d_f);
  if (ier > 0) return ier;

  cudaMemcpy(f, d_f, N * sizeof(float _Complex), cudaMemcpyDeviceToHost);

  cufinufftf_destroy(plan);

  cudaFree(d_x);
  cudaFree(d_c);
  cudaFree(d_f);

  idx = 4 * N / 7;

  printf("f[%d] = %lf + %lfi\n", idx, crealf(f[idx]), cimagf(f[idx]));

  f0 = 0; // direct type-1 sum, to compare

  for (int j = 0; j < M; ++j) {
    f0 += c[j] * cexp(I * x[j] * (idx - N / 2));
  }

  printf("f0[%d] = %lf + %lfi\n", idx, crealf(f0), cimagf(f0));
  // docs-end: getting_started-main

  // Fail on a CUDA error, nonfinite output, or relative error above 1e-3.
  // The single-precision direct sum above is only good to about 1e-4.
  const float maxerr = 1e-3f;
  float Fmax         = 0.0f;
  for (int m = 0; m < N; ++m) {
    const float a = cabsf(f[m]);
    if (a > Fmax || !isfinite(a)) Fmax = a; // a NaN entry makes Fmax NaN
  }
  const float err = cabsf(f[idx] - f0) / Fmax;
  printf("getting_started: rel err in f[%d] is %.3g\n", idx, err);
  const bool ok = cudaGetLastError() == cudaSuccess && isfinite(Fmax) && err <= maxerr;

  free(x);
  free(c);
  free(f);

  return ok ? 0 : 1;
}
