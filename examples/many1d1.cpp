#include <finufft.h>

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
using namespace std;

static const double PI = 3.141592653589793238462643383279502884;

int main()
/* Example of calling the vectorized FINUFFT library from C++, using STL
   double complex vectors, with a math test.
   To compile, see README.  Usage: ./many1d1
*/
{
  // generate some random nonuniform points (x) and complex strengths (c)...
  // docs-start: many1d1-data
  int ntrans = 3;   // how many stacked transforms to do
  int M      = 1e6; // nonuniform points (same for all transforms)
  vector<double> x(M);
  vector<complex<double>> c(M * ntrans);
  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-PI, PI), u1(-1.0, 1.0);
  generate(x.begin(), x.end(), [&] { return upi(rng); }); // uniform random in [-pi,pi)
  generate(c.begin(), c.end(),
           [&] { return complex<double>{u1(rng), u1(rng)}; }); // all ntrans vectors
  // docs-end: many1d1-data

  // allocate output array for the Fourier modes...
  // docs-start: many1d1-alloc
  int N = 1e6; // number of modes (same for all transforms)
  vector<complex<double>> F(N * ntrans);
  // docs-end: many1d1-alloc

  // call the NUFFT (with iflag=+1): note pointers (not STL vecs) passed...
  // docs-start: many1d1-call
  double tol = 1e-9; // desired accuracy
  int ier = finufft1d1many(ntrans, M, x.data(), c.data(), +1, tol, N, F.data(), nullptr);
  // docs-end: many1d1-call

  int k   = 142519;       // check the answer just for this mode...
  int trans = ntrans - 1; // ...in this transform
  assert(k >= -double(N) / 2 && k < double(N) / 2);
  complex<double> I     = complex<double>(0.0, 1.0);       // the imaginary unit
  complex<double> Ftest = complex<double>(0, 0);           // do the naive calc...
  for (int j = 0; j < M; ++j)
    Ftest += c[j + M * trans] * exp(I * double(k) * x[j]); // c from transform # trans
  double Fmax = 0.0; // compute inf norm of F for transform # trans
  for (int m = 0; m < N; ++m) {
    double aF = abs(F[m + N * trans]);
    if (aF > Fmax) Fmax = aF;
  }
  int kout   = k + N / 2 + N * trans; // output index, freq mode k, transform # trans
  double err = abs(F[kout] - Ftest) / Fmax;
  printf("1D type-1 double-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n", ier, k,
         err);
  if (!(err <= 10 * tol)) return 1;
  return ier;
}
