// this is all you must include to access finufft from C...
#include <finufft.h>

// also needed for this example...
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
/* Simple example of calling the FINUFFT library from C, using C complex type,
   with a math test. Double-precision. C99 style. opts is struct not ptr to it.

   Compile with:
   gcc -fopenmp example1d1c.c -I../include ../lib-static/libfinufft.a -o example1d1c
   -lfftw3 -lfftw3_omp -lm -lstdc++ or if you have built a single-core version: gcc
   example1d1c.c -I../include ../lib-static/libfinufft.a -o example1d1c -lfftw3 -lm
   -lstdc++

   Usage: ./example1d1c
*/
{
  int M      = 1e6;  // number of nonuniform points
  int N      = 1e6;  // number of modes
  double tol = 1e-9; // desired accuracy

  // generate some random nonuniform points (x) and complex strengths (c):
  double *x         = (double *)malloc(sizeof(double) * M);
  double complex *c = (double complex *)malloc(sizeof(double complex) * M);
  for (int j = 0; j < M; ++j) {
    x[j] = M_PI * (2 * ((double)rand() / RAND_MAX) - 1); // uniform random in [-pi,pi)
    c[j] =
        2 * ((double)rand() / RAND_MAX) - 1 + I * (2 * ((double)rand() / RAND_MAX) - 1);
  }
  // allocate complex output array for the Fourier modes
  double complex *F = (double complex *)malloc(sizeof(double complex) * N);

  finufft_opts opts;           // opts struct (not ptr)
  finufft_default_opts(&opts); // set default opts (must do this)
  opts.debug = 2;              // show how to override a default
  // opts.upsampfac = 1.25;              // other opts...

  // call the NUFFT (with iflag=+1), passing pointers...
  int ier = finufft1d1(M, x, c, +1, tol, N, F, &opts);

  int k = 142519;                       // check the answer just for this mode...
  assert(k >= -(double)N / 2 && k < (double)N / 2);
  double complex Ftest = 0.0 + 0.0 * I; // defined in complex.h (I too)
  for (int j = 0; j < M; ++j) Ftest += c[j] * cexp(I * (double)k * x[j]);
  double Fmax = 0.0;                    // compute inf norm of F
  for (int m = 0; m < N; ++m) {
    double aF = cabs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }
  int kout   = k + N / 2; // index in output array for freq mode k
  double err = cabs(F[kout] - Ftest) / Fmax;
  printf("1D type 1 NUFFT done. ier=%d, err in F[%d] rel to max(F) is %.3g\n", ier, k,
         err);

  free(x);
  free(c);
  free(F);
  return ier;
}
