/* Example of calling the cuFINUFFT guru interface from C, using C complex
   type, with a math test. Single precision, 1D type 1.
   To compile, see README in this directory. Usage: ./getting_started
*/
// docs-start: getting_started-includes
#include <cufinufft.h>

#include <complex.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static const double PI = 3.141592653589793238462643383279502884;
// docs-end: getting_started-includes

int main(void) {
  // points in [-pi,pi), strengths in [-1,1]^2, on the CPU
  // docs-start: getting_started-data
  const int M      = 100000; // number of nonuniform points
  float *x         = malloc(M * sizeof(float));
  float complex *c = malloc(M * sizeof(float complex));
  srand(0);
  for (int j = 0; j < M; ++j) {
    x[j] = PI * (2 * (float)rand() / RAND_MAX - 1);
    c[j] = (2 * (float)rand() / RAND_MAX - 1) + I * (2 * (float)rand() / RAND_MAX - 1);
  }
  // docs-end: getting_started-data

  // copy x, c to the GPU (float complex has the same layout as cuFloatComplex)
  // docs-start: getting_started-device
  float *d_x;
  cuFloatComplex *d_c;
  cudaMalloc((void **)&d_x, M * sizeof(float));
  cudaMalloc((void **)&d_c, M * sizeof(cuFloatComplex));
  cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, M * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
  // docs-end: getting_started-device

  // allocate output arrays for the Fourier modes, on the CPU and the GPU
  // docs-start: getting_started-alloc
  const int N      = 10000; // number of modes
  float complex *f = malloc(N * sizeof(float complex));
  cuFloatComplex *d_f;
  cudaMalloc((void **)&d_f, N * sizeof(cuFloatComplex));
  // docs-end: getting_started-alloc

  // 1) make a plan: type 1, 1D, iflag +1, 1 transform
  // docs-start: getting_started-makeplan
  int64_t modes[1] = {N}; // number of modes in each dimension
  const float tol  = 1e-4f;
  cufinufftf_plan plan;
  int ier = cufinufftf_makeplan(1, 1, modes, +1, 1, tol, &plan, NULL);
  // docs-end: getting_started-makeplan
  if (ier) return ier;
  // 2) send in the points
  // docs-start: getting_started-setpts
  ier = cufinufftf_setpts(plan, M, d_x, NULL, NULL, 0, NULL, NULL, NULL);
  // docs-end: getting_started-setpts
  if (ier) return ier;
  // 3) do the transform
  // docs-start: getting_started-execute
  ier = cufinufftf_execute(plan, d_c, d_f);
  // docs-end: getting_started-execute
  if (ier) return ier;
  // copy f to the CPU, 4) free the plan, and free the GPU arrays
  // docs-start: getting_started-destroy
  cudaMemcpy(f, d_f, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
  cufinufftf_destroy(plan);
  cudaFree(d_x);
  cudaFree(d_c);
  cudaFree(d_f);
  // docs-end: getting_started-destroy

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
