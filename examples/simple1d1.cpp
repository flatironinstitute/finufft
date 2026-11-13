// docs-start: simple1d1-includes
// this is all you must include for the finufft lib...
#include <finufft.h>

// also used in this example...
#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
using namespace std;

static const double PI = 3.141592653589793238462643383279502884;
// docs-end: simple1d1-includes

int main()
/* Example of calling the FINUFFT library from C++, using STL
   double complex vectors, with a math test.
   Double-precision version (see simple1d1f for single-precision).
   To compile, see README in this directory.
   Also see ../docs/cex.rst or online documentation.
   Usage: ./simple1d1
*/
{
  // generate some random nonuniform points (x) and complex strengths (c)...
  // docs-start: simple1d1-data
  int M = 1e7; // number of nonuniform points
  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-PI, PI), u1(-1.0, 1.0);
  vector<double> x(M);
  vector<complex<double>> c(M);
  generate(x.begin(), x.end(), [&] { return upi(rng); }); // uniform random in [-pi,pi)
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });
  // docs-end: simple1d1-data

  // allocate output array for the Fourier modes...
  // docs-start: simple1d1-alloc
  int N = 1e6; // number of modes
  vector<complex<double>> F(N);
  // docs-end: simple1d1-alloc

  // call the NUFFT (with iflag=+1): note pointers (not STL vecs) passed...
  // docs-start: simple1d1-call
  double acc = 1e-9; // desired accuracy
  int ier    = finufft1d1(M, x.data(), c.data(), +1, acc, N, F.data(), nullptr);
  // docs-end: simple1d1-call

  int k      = 142519; // check the answer just for this mode frequency...
  assert(k >= -double(N) / 2 && k < double(N) / 2);
  complex<double> I     = complex<double>(0.0, 1.0); // the imaginary unit
  complex<double> Ftest = complex<double>(0, 0);
  for (int j = 0; j < M; ++j) Ftest += c[j] * exp(I * double(k) * x[j]);
  double Fmax = 0.0; // compute inf norm of F
  for (int m = 0; m < N; ++m) {
    double aF = abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }
  int kout   = k + N / 2; // index in output array for freq mode k
  double err = abs(F[kout] - Ftest) / Fmax;
  printf("1D type-1 double-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n", ier, k,
         err);
  if (!(err <= 10 * acc)) return 1;
  return ier;
}
