#include <finufft.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
using namespace std;

constexpr double pi = 3.14159265358979323846;

int main()
/* Example of calling the vectorized FINUFFT library from C++, using STL
   double complex vectors, with a math test.
*/
{
  constexpr int ntrans = 3;    // how many stacked transforms to do
  constexpr int M      = 1e6;  // nonuniform points (same for all transforms)
  constexpr int N      = 1e6;  // number of modes (same for all transforms)
  constexpr double tol = 1e-9; // desired accuracy
  complex<double> I    = complex<double>(0.0, 1.0); // the imaginary unit

  // generate some random nonuniform points (x) and complex strengths (c)...
  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);
  vector<double> x(M);
  vector<complex<double>> c(M * ntrans), F(N * ntrans); // F: output Fourier modes
  // docs-start: many1d1
  generate(x.begin(), x.end(), [&] { return upi(rng); });
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });

  // call the NUFFT (with iflag=+1): note pointers (not STL vecs) passed...
  int ier = finufft1d1many(ntrans, M, &x[0], &c[0], +1, tol, N, &F[0], NULL);
  // docs-end: many1d1
  if (ier) return ier;

  constexpr int k     = 142519;     // check the answer just for this mode...
  constexpr int trans = ntrans - 1; // ...in this transform
  static_assert(k >= -(double)N / 2 && k < (double)N / 2);
  complex<double> Ftest(0.0, 0.0);  // do the naive calc...
  for (int j = 0; j < M; ++j)
    Ftest += c[j + M * trans] * exp(I * double(k) * x[j]); // c from transform # trans
  double Fmax = 0.0; // compute inf norm of F for transform # trans
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
