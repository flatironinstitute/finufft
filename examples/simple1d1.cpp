/* Example of calling the FINUFFT library from C++, using STL double complex
   vectors, with a math test. The code between the docs-start/docs-end markers
   is shown in the online documentation (../docs/cex.rst).
   Double-precision version (see simple1d1f for single-precision).
*/
#include <cmath>
#include <cstdio>
// docs-start: simple1d1-includes
#include "finufft.h"
#include <complex>
#include <stdlib.h>
#include <vector>
// docs-end: simple1d1-includes
using namespace std;

constexpr double PI = 3.141592653589793238462643383279502884;

int main() {
  // docs-start: simple1d1-data
  int M = 1e7; // number of nonuniform points
  vector<double> x(M);
  vector<complex<double>> c(M);
  complex<double> I = complex<double>(0.0, 1.0);     // the imaginary unit
  for (int j = 0; j < M; ++j) {
    x[j] = PI * (2 * (double)rand() / RAND_MAX - 1); // uniform random in [-pi,pi)
    c[j] = 2 * (double)rand() / RAND_MAX - 1 + I * (2 * (double)rand() / RAND_MAX - 1);
  }
  // docs-end: simple1d1-data
  // docs-start: simple1d1-modes
  int N = 1e6; // number of output modes
  vector<complex<double>> F(N);
  // docs-end: simple1d1-modes
  // docs-start: simple1d1-call
  int ier = finufft1d1(M, &x[0], &c[0], +1, 1e-9, N, &F[0], NULL);
  // docs-end: simple1d1-call
  if (ier) return ier;

  // math test, after the documented code:
  constexpr double tol = 1e-9;   // the tolerance requested above
  constexpr int k      = 142519; // check the answer just for this mode, in [-N/2,N/2)
  complex<double> Ftest(0.0, 0.0);
  for (int j = 0; j < M; ++j) Ftest += c[j] * exp(I * double(k) * x[j]);
  double Fmax = 0.0; // compute inf norm of F
  bool Ffin   = true;
  for (const auto &Fm : F) {
    const double a = abs(Fm);
    Ffin &= std::isfinite(a);
    if (a > Fmax) Fmax = a;
  }
  const auto err = abs(F[k + N / 2] - Ftest) / Fmax; // k + N/2: index of freq mode k
  if (!(Ffin && Fmax > 0.0 && err < 10 * tol)) {
    fprintf(stderr, "FAILED: F non-finite or rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  printf("rel err %.3g\n", err);
  return 0;
}
