// docs-start: simple1d1c-includes
// this is all you must include to access finufft from C...
#include <finufft.h>

// also needed for this example...
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static const double PI = 3.141592653589793238462643383279502884;
// docs-end: simple1d1c-includes

int main()
/* Simple example of calling the FINUFFT library from C, using C complex type,
   with a math test. Double-precision. C99 style. opts is struct not ptr to it.
   To compile see README. Usage: ./simple1d1c
*/
{
  // generate some random nonuniform points (x) and complex strengths (c):
  // docs-start: simple1d1c-data
  int M             = 1e6; // number of nonuniform points
  double *x         = (double *)malloc(sizeof(double) * M);
  double complex *c = (double complex *)malloc(sizeof(double complex) * M);
  for (int j = 0; j < M; ++j) {
    x[j] = PI * (2 * ((double)rand() / RAND_MAX) - 1); // uniform random in [-pi,pi)
    c[j] =
        2 * ((double)rand() / RAND_MAX) - 1 + I * (2 * ((double)rand() / RAND_MAX) - 1);
  }
  // docs-end: simple1d1c-data

  // allocate complex output array for the Fourier modes
  // docs-start: simple1d1c-alloc
  int N             = 1e6; // number of modes
  double complex *F = (double complex *)malloc(sizeof(double complex) * N);
  // docs-end: simple1d1c-alloc

  // call the NUFFT (with iflag=+1), passing pointers...
  // docs-start: simple1d1c-call
  finufft_opts opts;           // opts struct (not ptr)
  finufft_default_opts(&opts); // set default opts (must do this)
  opts.debug = 2;              // show how to override a default
  double tol = 1e-9;           // desired accuracy
  int ier    = finufft1d1(M, x, c, +1, tol, N, F, &opts);
  // docs-end: simple1d1c-call
  if (ier) return ier;

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
  if (!(err <= 10 * tol)) {
    fprintf(stderr, "FAILED: rel err %.3g in F[%d]\n", err, k);
    return 1;
  }

  // docs-start: simple1d1c-free
  free(x);
  free(c);
  free(F);
  // docs-end: simple1d1c-free
  return ier;
}
