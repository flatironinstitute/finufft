// docs-start: quick-start
// this is all you must include for the finufft lib...
#include <finufft.h>

// also used in this example...
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
using namespace std;

constexpr double pi = 3.14159265358979323846;

int main()
/* Example of calling the FINUFFT library from C++, using STL
   double complex vectors, with a math test.
   Double-precision version (see simple1d1f for single-precision).
   Also see ../docs/cex.rst or online documentation.
*/
{
  // docs-end: quick-start
  constexpr int M      = 1e7;  // number of nonuniform points
  constexpr int N      = 1e6;  // number of modes
  constexpr double tol = 1e-9; // desired accuracy
  // docs-start: walkthrough
  finufft_opts opts;                 // opts is a plain struct
  finufft_default_opts(&opts);
  const complex<double> I(0.0, 1.0); // the imaginary unit
  // docs-end: walkthrough
  // generate some random nonuniform points (x) and complex strengths (c)...
  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);
  vector<double> x(M);
  vector<complex<double>> c(M), F(N);
  generate(x.begin(), x.end(), [&] { return upi(rng); });
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });

  // docs-start: transform
  // call the NUFFT (with iflag=+1): note pointers (not STL vecs) passed...
  int ier = finufft1d1(M, &x[0], &c[0], +1, tol, N, &F[0], &opts);
  // docs-end: transform
  if (ier) return ier;
  constexpr int k = 142519; // check the answer just for this mode frequency...
  static_assert(k >= -(double)N / 2 && k < (double)N / 2);
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
