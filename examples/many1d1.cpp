/* Example of calling the vectorized FINUFFT library from C++, using STL
   double complex vectors, with a math test.
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
  // docs-start: many1d1
  int ntrans = 10;                               // how many transforms
  int M      = 1e7;                              // number of nonuniform points
  vector<double> x(M);
  vector<complex<double>> c(M * ntrans);         // ntrans strength vectors
  complex<double> I = complex<double>(0.0, 1.0); // the imaginary unit
  for (int j = 0; j < M; ++j) x[j] = PI * (2 * (double)rand() / RAND_MAX - 1);
  for (int j = 0; j < M * ntrans; ++j)           // fill all ntrans vectors...
    c[j] = 2 * (double)rand() / RAND_MAX - 1 + I * (2 * (double)rand() / RAND_MAX - 1);
  int N = 1e6;                                   // number of output modes
  vector<complex<double>> F(N * ntrans);         // ntrans output vectors
  // NULL: default opts
  int ier = finufft1d1many(ntrans, M, &x[0], &c[0], +1, 1e-9, N, &F[0], NULL);
  // docs-end: many1d1
  if (ier) return ier;

  // math test, after the documented code:
  constexpr double tol = 1e-9;       // the tolerance requested above
  constexpr int k      = 142519;     // check the answer just for this mode, in [-N/2,N/2)
  const int trans      = ntrans - 1; // ...in this transform
  complex<double> Ftest(0.0, 0.0);
  for (int j = 0; j < M; ++j)
    Ftest += c[j + M * trans] * exp(I * double(k) * x[j]); // c from transform # trans
  double Fmax = 0.0;                                       // compute inf norm of F
  bool Ffin   = true;
  for (const auto &Fm : F) {
    const double a = abs(Fm);
    Ffin &= std::isfinite(a);
    if (a > Fmax) Fmax = a;
  }
  const auto err = abs(F[k + N / 2 + N * trans] - Ftest) / Fmax; // output index of k
  if (!(Ffin && Fmax > 0.0 && err < 10 * tol)) {
    fprintf(stderr, "FAILED: F non-finite or rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  printf("rel err %.3g\n", err);
  return 0;
}
