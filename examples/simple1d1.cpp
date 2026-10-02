// docs-start: quick-start
// this is all you must include for the finufft lib...
#include <finufft.h>

// also used in this example...
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
// docs-end: quick-start

int main()
/* Example of calling the FINUFFT library from C++, using STL
   double complex vectors, with a math test.
   Double-precision version (see simple1d1f for single-precision).
   To compile, see README in this directory.
   Also see ../docs/cex.rst or online documentation.
   Usage: ./simple1d1
*/
{
  // docs-start: walkthrough
  constexpr int M      = 1e6;                    // number of nonuniform points
  constexpr int N      = 1e6;                    // number of modes
  constexpr double acc = 1e-9;                   // desired accuracy
  finufft_opts opts;                             // opts is a plain struct
  finufft_default_opts(&opts);
  complex<double> I = complex<double>(0.0, 1.0); // the imaginary unit
  // docs-end: walkthrough
  // docs-start: declare-fill
  // generate some random nonuniform points (x) and complex strengths (c)...
  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);
  vector<double> x(M);
  vector<complex<double>> c(M), F(N);
  generate(x.begin(), x.end(), [&] { return upi(rng); });
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });
  // docs-end: declare-fill

  // docs-start: transform
  // call the NUFFT (with iflag=+1): note pointers (not STL vecs) passed...
  int ier = finufft1d1(M, &x[0], &c[0], +1, acc, N, &F[0], &opts);
  // docs-end: transform
  if (ier) return ier;
  constexpr int k = 142519; // check the answer just for this mode frequency...
  assert(k >= -(double)N / 2 && k < (double)N / 2);
  vector<complex<double>> terms(M);
  transform(c.begin(), c.end(), x.begin(), terms.begin(),
            [&](auto cj, auto xj) { return cj * exp(I * double(k) * xj); });
  const auto Ftest = reduce(terms.begin(), terms.end());
  const auto Fmax = abs(
      *max_element(F.begin(), F.end(), [](auto a, auto b) { return abs(a) < abs(b); }));
  const auto err = abs(F[k + N / 2] - Ftest) / Fmax; // k + N/2: index of freq mode k
  printf("1D type-1 double-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n", ier, k,
         err);
  return !(err < 10 * acc);
}
