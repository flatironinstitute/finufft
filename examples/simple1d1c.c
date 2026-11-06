/* Simple example of calling the FINUFFT library from C, using C complex type,
   with a math test. Double-precision. C99 style. opts is struct not ptr to it.
   To build, see docs/cex.rst. Usage: ./simple1d1c
   The code between the docs-start and docs-end markers is shown in docs/cex.rst
   (not int main), so keep it short.
*/
#define _USE_MATH_DEFINES // M_PI on MSVC
#include <math.h>
#include <stdio.h>

// docs-start: simple1d1c-includes
#include <complex.h>
#include <finufft.h>
#include <stdlib.h>
// docs-end: simple1d1c-includes

int main() {
  // docs-start: simple1d1c-body
  int M             = 1e7; // number of nonuniform points
  double *x         = (double *)malloc(sizeof(double) * M);
  double complex *c = (double complex *)malloc(sizeof(double complex) * M);
  for (int j = 0; j < M; ++j) {
    x[j] = M_PI * (2 * (double)rand() / RAND_MAX - 1); // uniform random in [-pi,pi)
    c[j] = 2 * (double)rand() / RAND_MAX - 1 + I * (2 * (double)rand() / RAND_MAX - 1);
  }
  int N             = 1e6;     // number of modes
  double complex *F = (double complex *)malloc(sizeof(double complex) * N);
  finufft_opts opts;           // make an opts struct
  finufft_default_opts(&opts); // set default opts (must do this)
  opts.debug = 2;              // more debug/timing to stdout
  int ier    = finufft1d1(M, x, c, +1, 1e-9, N, F, &opts);

  // (now do something with F here!...)
  // docs-end: simple1d1c-body

  if (ier) return ier;           // no valid output to read

  int k                = 142519; // check the answer just for this mode...
  double complex Ftest = 0.0 + 0.0 * I;
  for (int j = 0; j < M; ++j) Ftest += c[j] * cexp(I * (double)k * x[j]);
  double Fmax = 0.0; // compute inf norm of F
  int finite  = 1;   // track non-finite elements
  for (int m = 0; m < N; ++m) {
    double aF = cabs(F[m]);
    if (!isfinite(aF)) finite = 0;
    if (aF > Fmax) Fmax = aF;
  }
  double err = cabs(F[k + N / 2] - Ftest) / Fmax; // F[k + N/2] is mode k
  if (!finite || !(err < 10 * 1e-9)) {
    fprintf(stderr, "FAILED: rel err %.3g in F[%d], or F not finite\n", err, k);
    return 1;
  }
  printf("rel err in F[%d] is %.3g\n", k, err);

  // docs-start: simple1d1c-free
  free(x);
  free(c);
  free(F);
  // docs-end: simple1d1c-free
  return 0;
}
