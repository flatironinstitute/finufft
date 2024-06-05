// this is all you must include...
#include <finufft.h>

// also needed for this example...
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
/* Simple example of calling the FINUFFT library from C, using C complex type,
   with a math test. Single-precision version. C99 style. opts is a struct.

   Compile with:
   gcc -fopenmp example1d1cf.c -I../include ../lib-static/libfinufft.a -o example1d1cf
   -lfftw3f -lfftw3f_omp -lm -lstdc++ or if you have built a single-core version: gcc
   example1d1cf.c -I../include ../lib-static/libfinufft.a -o example1d1cf -lfftw3f -lm
   -lstdc++

   Usage: ./example1d1cf
*/
{
  int M     = 1e5;  // number of nonuniform points
  int N     = 1e5;  // number of modes (NB if too large lose acc in 1d)
  float tol = 1e-3; // desired accuracy

  // generate some random nonuniform points (x) and complex strengths (c):
  float *x         = (float *)malloc(sizeof(float) * M);
  float complex *c = (float complex *)malloc(sizeof(float complex) * M);
  for (int j = 0; j < M; ++j) {
    x[j] = M_PI * (2 * ((float)rand() / RAND_MAX) - 1); // uniform random in [-pi,pi)
    c[j] = 2 * ((float)rand() / RAND_MAX) - 1 + I * (2 * ((float)rand() / RAND_MAX) - 1);
  }
  // allocate complex output array for the Fourier modes
  float complex *F = (float complex *)malloc(sizeof(float complex) * N);

  finufft_opts opts;            // opts struct (not ptr)
  finufftf_default_opts(&opts); // set default opts (must do this)
  opts.debug = 2;               // show how to override a default
  // opts.upsampfac = 1.25;                 // other opts...

  // call the NUFFT (with iflag=+1), passing pointers...
  int ier = finufftf1d1(M, x, c, +1, tol, N, F, &opts);

  int k = 14251;                         // check the answer just for this mode...
  assert(k >= -(double)N / 2 && k < (double)N / 2);
  float complex Ftest = 0.0f + 0.0f * I; // defined in complex.h (I too)
  for (int j = 0; j < M; ++j) Ftest += c[j] * cexpf(I * (float)k * x[j]);
  float Fmax = 0.0;                      // compute inf norm of F
  for (int m = 0; m < N; ++m) {
    float aF = cabsf(F[m]);
    if (aF > Fmax) Fmax = aF;
  }
  int kout  = k + N / 2; // index in output array for freq mode k
  float err = cabsf(F[kout] - Ftest) / Fmax;
  printf("1D type 1 NUFFT, single-prec. ier=%d, err in F[%d] rel to max(F) is %.3g\n",
         ier, k, err);

  free(x);
  free(c);
  free(F);
  return ier;
}
