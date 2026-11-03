/* Simple 2D type-1 example of calling the FINUFFT library from C++, using plain
   arrays of C++ complex numbers, with a math test. Double precision version.
   The code between the docs-start/docs-end markers is shown in ../docs/cex.rst.
*/
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <finufft.h>
#include <vector>
using namespace std;

constexpr double PI = 3.141592653589793238462643383279502884;

int main() {
  complex<double> I(0.0, 1.0); // the imaginary unit
  // docs-start: simple2d1-data
  int M = 1e7; // number of nonuniform points
  vector<double> x(M), y(M);
  vector<complex<double>> c(M);
  for (int j = 0; j < M; ++j) {
    x[j] = PI * (2 * (double)rand() / RAND_MAX - 1);
    y[j] = PI * (2 * (double)rand() / RAND_MAX - 1);
    c[j] = 2 * (double)rand() / RAND_MAX - 1 + I * (2 * (double)rand() / RAND_MAX - 1);
  }
  // docs-end: simple2d1-data
  // docs-start: simple2d1-call
  int N1 = 1000, N2 = 2000;
  vector<complex<double>> F(N1 * N2);
  int ier = finufft2d1(M, &x[0], &y[0], &c[0], +1, 1e-6, N1, N2, &F[0], NULL);
  // docs-end: simple2d1-call
  if (ier) return ier;

  // math test, after the documented code:
  constexpr double tol = 1e-6;             // the tolerance requested above
  const int k1         = round(0.45 * N1); // check the answer for mode frequency (k1,k2)
  const int k2         = round(-0.35 * N2);
  complex<double> Ftest(0.0, 0.0);
  for (int j = 0; j < M; ++j)
    Ftest += c[j] * exp(I * (double(k1) * x[j] + double(k2) * y[j]));
  double Fmax = 0.0; // compute inf norm of F
  bool Ffin   = true;
  for (const auto &Fm : F) {
    const double a = abs(Fm);
    Ffin &= std::isfinite(a);
    if (a > Fmax) Fmax = a;
  }
  // index in output array for this frequency pair (k1,k2)
  const auto err = abs(F[k1 + N1 / 2 + (k2 + N2 / 2) * N1] - Ftest) / Fmax;
  if (!(Ffin && Fmax > 0.0 && err < 10 * tol)) {
    fprintf(stderr, "FAILED: F non-finite or rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  printf("rel err %.3g\n", err);
  return 0;
}
