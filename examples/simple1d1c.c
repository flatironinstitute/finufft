/* Simple example of calling the FINUFFT library from C, using C complex type,
   with a math test. Double-precision. C99 style. opts is struct not ptr to it.
   To build, see docs/cex.rst. Usage: ./simple1d1c
*/

// docs-start: quick-start-c
// this is all you must include to access finufft from C...
#include <finufft.h>

// also needed for this example...
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static const double PI = 3.141592653589793238462643383279502884;
// docs-end: quick-start-c

int main() {
  int M      = 1e6;  // number of nonuniform points
  int N      = 1e6;  // number of modes
  double tol = 1e-9; // desired accuracy

  // generate some random nonuniform points (x) and complex strengths (c):
  double *x         = (double *)malloc(sizeof(double) * M);
  double complex *c = (double complex *)malloc(sizeof(double complex) * M);
  for (int j = 0; j < M; ++j) {
    x[j] = PI * (2 * ((double)rand() / RAND_MAX) - 1); // uniform random in [-pi,pi)
    c[j] =
        2 * ((double)rand() / RAND_MAX) - 1 + I * (2 * ((double)rand() / RAND_MAX) - 1);
  }
  // allocate complex output array for the Fourier modes
  double complex *F = (double complex *)malloc(sizeof(double complex) * N);

  // docs-start: call
  finufft_opts opts;           // opts struct (not ptr)
  finufft_default_opts(&opts); // set default opts (must do this)

  // call the NUFFT (with iflag=+1), passing pointers...
  int ier = finufft1d1(M, x, c, +1, tol, N, F, &opts);
  // docs-end: call
  if (ier) return ier; // no valid output to read

  // (now do something with F here!...)

  int k                = 142519;        // check the answer just for this mode...
  double complex Ftest = 0.0 + 0.0 * I; // defined in complex.h (I too)
  for (int j = 0; j < M; ++j) Ftest += c[j] * cexp(I * (double)k * x[j]);
  double Fmax = 0.0;                    // compute inf norm of F
  int finite  = 1;                      // track non-finite elements
  for (int m = 0; m < N; ++m) {
    double aF = cabs(F[m]);
    if (!isfinite(aF)) finite = 0;
    if (aF > Fmax) Fmax = aF;
  }
  int kout   = k + N / 2; // index in output array for freq mode k
  double err = cabs(F[kout] - Ftest) / Fmax;
  if (!finite || !(err < 10 * tol)) {
    fprintf(stderr, "FAILED: rel err %.3g in F[%d], or F not finite\n", err, k);
    return 1;
  }
  printf("rel err in F[%d] is %.3g\n", k, err);

  // docs-start: destroy
  // free the memory
  free(x);
  free(c);
  free(F);
  // docs-end: destroy
  return 0;
}
