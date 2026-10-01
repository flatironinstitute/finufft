// docs-start: many1d1
#include <finufft.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>
using namespace std;

constexpr double pi = 3.14159265358979323846;

int main()
/* Example of calling the vectorized FINUFFT library from C++, using STL
   double complex vectors, with a math test.
   To compile, see README.  Usage: ./many1d1
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
  generate(x.begin(), x.end(), [&] { return upi(rng); });
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });

  // call the NUFFT (with iflag=+1): note pointers (not STL vecs) passed...
  int ier = finufft1d1many(ntrans, M, &x[0], &c[0], +1, tol, N, &F[0], NULL);
  // docs-end: many1d1
  if (ier) return ier;

  constexpr int k     = 142519;     // check the answer just for this mode...
  constexpr int trans = ntrans - 1; // ...in this transform
  assert(k >= -(double)N / 2 && k < (double)N / 2);
  vector<complex<double>> terms(M); // naive calc, c from transform # trans
  transform(x.begin(), x.end(), c.begin() + M * trans, terms.begin(),
            [&](auto xj, auto cj) { return cj * exp(I * double(k) * xj); });
  const auto Ftest            = reduce(terms.begin(), terms.end());
  const auto Ft               = F.begin() + N * trans;
  const auto Fmax =
      abs(*max_element(Ft, Ft + N, [](auto a, auto b) { return abs(a) < abs(b); }));
  const auto err = abs(F[k + N / 2 + N * trans] - Ftest) / Fmax; // output index of k
  printf("1D type-1 double-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n", ier, k,
         err);
  return !(err < 10 * tol);
}
