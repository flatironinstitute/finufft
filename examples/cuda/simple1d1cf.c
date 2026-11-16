/* Example of calling the cuFINUFFT simple interface from C, using C complex
   type, with a math test. Single precision, 1D type 1. GPU version of
   examples/simple1d1cf.c. Barbone 11/16/26.
   To compile, see README in this directory. Usage: ./simple1d1cf
*/
// docs-start: simple1d1cf-includes
#include <cufinufft.h>

#include <complex.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static const double PI = 3.141592653589793238462643383279502884;
// docs-end: simple1d1cf-includes

int main(void) {
  const int M      = 100000; // number of nonuniform points
  const int N      = 10000;  // number of modes
  const float tol  = 1e-3f;  // desired accuracy

  // points in [-pi,pi), strengths in [-1,1]^2, and room for the modes, on the CPU
  float *x         = malloc(M * sizeof(float));
  float complex *c = malloc(M * sizeof(float complex));
  float complex *f = malloc(N * sizeof(float complex));
  srand(0);
  for (int j = 0; j < M; ++j) {
    x[j] = PI * (2 * (float)rand() / RAND_MAX - 1);
    c[j] = (2 * (float)rand() / RAND_MAX - 1) + I * (2 * (float)rand() / RAND_MAX - 1);
  }

  // copy x, c to the GPU and make room for f (float complex has the same layout as
  // cuFloatComplex)
  // docs-start: simple1d1cf-device
  float *d_x;
  cuFloatComplex *d_c, *d_f;
  cudaMalloc((void **)&d_x, M * sizeof(float));
  cudaMalloc((void **)&d_c, M * sizeof(cuFloatComplex));
  cudaMalloc((void **)&d_f, N * sizeof(cuFloatComplex));
  cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, M * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
  // docs-end: simple1d1cf-device
  // the cufinufftf prefix selects single precision
  // docs-start: simple1d1cf-call
  int ier = cufinufftf1d1(M, d_x, d_c, +1, tol, N, d_f, NULL);
  // docs-end: simple1d1cf-call
  // copy f back to the CPU and free the GPU arrays
  // docs-start: simple1d1cf-copy
  cudaMemcpy(f, d_f, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_c);
  cudaFree(d_f);
  // docs-end: simple1d1cf-copy
  if (ier) return ier;

  // check the answer just for mode k in [-N/2,N/2), by a direct sum in double
  int k                = 1425;
  double complex Ftest = 0;
  for (int j = 0; j < M; ++j) Ftest += c[j] * cexp(I * k * (double)x[j]);
  float Fmax = 0; // inf norm of f
  for (int m = 0; m < N; ++m) {
    float a = cabsf(f[m]);
    if (a > Fmax) Fmax = a;
  }
  double err = cabs(f[k + N / 2] - Ftest) / Fmax; // k + N/2: index of freq mode k

  free(x);
  free(c);
  free(f);
  if (!(err <= 10 * tol) || cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "FAILED: rel err %.3g > %.3g\n", err, 10.0 * tol);
    return 1;
  }
  printf("rel err %.3g\n", err);
  return 0;
}
