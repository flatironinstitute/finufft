#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cufinufft.h>

#include <complex.h>

int test_float(int M, int N) {
  // Size of the grid as an array.
  int64_t modes[1] = {N};

  // Host pointers: frequencies (x), coefficients (c), and output (f).
  float *x;
  float _Complex *c;
  float _Complex *f;

  // Device pointers.
  float *d_x;
  cuFloatComplex *d_c, *d_f;

  // Store cufinufft plan.
  cufinufftf_plan plan;

  // Manual calculation at a single point idx.
  int idx;
  float _Complex f0;

  // Allocate the host arrays.
  x = (float *)malloc(M * sizeof(float));
  c = (float _Complex *)malloc(M * sizeof(float _Complex));
  f = (float _Complex *)malloc(N * sizeof(float _Complex));

  // Fill with random numbers. Frequencies must be in the interval [-pi, pi]
  // while strengths can be any value.
  srand(0);

  for (int j = 0; j < M; ++j) {
    x[j] = 2 * M_PI * (((float)rand()) / RAND_MAX - 1);
    c[j] =
        (2 * ((float)rand()) / RAND_MAX - 1) + I * (2 * ((float)rand()) / RAND_MAX - 1);
  }

  // Allocate the device arrays and copy the x and c arrays.
  cudaMalloc((void **)&d_x, M * sizeof(float));
  cudaMalloc((void **)&d_c, M * sizeof(float _Complex));
  cudaMalloc((void **)&d_f, N * sizeof(float _Complex));

  cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, M * sizeof(float _Complex), cudaMemcpyHostToDevice);

  // Make the cufinufft plan for a 1D type-1 transform with six digits of
  // tolerance.
  cufinufftf_makeplan(1, 1, modes, 1, 1, 1e-6, &plan, NULL);

  // Set the frequencies of the nonuniform points.
  cufinufftf_setpts(plan, M, d_x, NULL, NULL, 0, NULL, NULL, NULL);

  // Actually execute the plan on the given coefficients and store the result
  // in the d_f array.
  cufinufftf_execute(plan, d_c, d_f);

  // Copy the result back onto the host.
  cudaMemcpy(f, d_f, N * sizeof(float _Complex), cudaMemcpyDeviceToHost);

  // Destroy the plan and free the device arrays after we're done.
  cufinufftf_destroy(plan);

  cudaFree(d_x);
  cudaFree(d_c);
  cudaFree(d_f);

  // Pick an index to check the result of the calculation.
  idx = 4 * N / 7;

  printf("f[%d] = %lf + %lfi\n", idx, crealf(f[idx]), cimagf(f[idx]));

  // Calculate the result manually using the formula for the type-1
  // transform.
  f0 = 0;

  for (int j = 0; j < M; ++j) {
    f0 += c[j] * cexp(I * x[j] * (idx - N / 2));
  }

  printf("f0[%d] = %lf + %lfi\n", idx, crealf(f0), cimagf(f0));

  // Finally free the host arrays.
  free(x);
  free(c);
  free(f);

  return 0;
}

int test_double(int M, int N) {
  // Size of the grid as an array.
  int64_t modes[1] = {N};

  // Host pointers: frequencies (x), coefficients (c), and output (f).
  double *x;
  double _Complex *c;
  double _Complex *f;

  // Device pointers.
  double *d_x;
  cuDoubleComplex *d_c, *d_f;

  // Store cufinufft plan.
  cufinufft_plan plan;

  // Manual calculation at a single point idx.
  int idx;
  double _Complex f0;

  // Allocate the host arrays.
  x = (double *)malloc(M * sizeof(double));
  c = (double _Complex *)malloc(M * sizeof(double _Complex));
  f = (double _Complex *)malloc(N * sizeof(double _Complex));

  // Fill with random numbers. Frequencies must be in the interval [-pi, pi]
  // while strengths can be any value.
  srand(0);

  for (int j = 0; j < M; ++j) {
    x[j] = 2 * M_PI * (((double)rand()) / RAND_MAX - 1);
    c[j] =
        (2 * ((double)rand()) / RAND_MAX - 1) + I * (2 * ((double)rand()) / RAND_MAX - 1);
  }

  // Allocate the device arrays and copy the x and c arrays.
  cudaMalloc((void **)&d_x, M * sizeof(double));
  cudaMalloc((void **)&d_c, M * sizeof(double _Complex));
  cudaMalloc((void **)&d_f, N * sizeof(double _Complex));

  cudaMemcpy(d_x, x, M * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, M * sizeof(double _Complex), cudaMemcpyHostToDevice);

  // Make the cufinufft plan for a 1D type-1 transform with six digits of
  // tolerance.
  cufinufft_makeplan(1, 1, modes, 1, 1, 1e-6, &plan, NULL);

  // Set the frequencies of the nonuniform points.
  cufinufft_setpts(plan, M, d_x, NULL, NULL, 0, NULL, NULL, NULL);

  // Actually execute the plan on the given coefficients and store the result
  // in the d_f array.
  cufinufft_execute(plan, d_c, d_f);

  // Copy the result back onto the host.
  cudaMemcpy(f, d_f, N * sizeof(double _Complex), cudaMemcpyDeviceToHost);

  // Destroy the plan and free the device arrays after we're done.
  cufinufft_destroy(plan);

  cudaFree(d_x);
  cudaFree(d_c);
  cudaFree(d_f);

  // Pick an index to check the result of the calculation.
  idx = 4 * N / 7;

  printf("f[%d] = %lf + %lfi\n", idx, crealf(f[idx]), cimagf(f[idx]));

  // Calculate the result manually using the formula for the type-1
  // transform.
  f0 = 0;

  for (int j = 0; j < M; ++j) {
    f0 += c[j] * cexp(I * x[j] * (idx - N / 2));
  }

  printf("f0[%d] = %lf + %lfi\n", idx, crealf(f0), cimagf(f0));

  // Finally free the host arrays.
  free(x);
  free(c);
  free(f);

  return 0;
}

int main() {
  // Problem size: number of nonuniform points (M) and grid size (N).
  const int M = 100, N = 200;
  int errf = test_float(M, N);
  int err  = test_double(M, N);

  return (err | errf);
}
