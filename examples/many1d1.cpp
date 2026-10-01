// docs-start: many1d1
#include <finufft.h>

#include <cassert>
#include <complex>
#include <cstdio>
#include <stdlib.h>
#include <vector>
using namespace std;

static const double PI = 3.141592653589793238462643383279502884;

int main()
/* Example of calling the vectorized FINUFFT library from C++, using STL
   double complex vectors, with a math test.
   To compile, see README.  Usage: ./many1d1
*/
{
  int ntrans         = 3;                // how many stacked transforms to do
  int M              = 1e6;              // nonuniform points (same for all transforms)
  int N              = 1e6;              // number of modes (same for all transforms)
  double tol         = 1e-9;             // desired accuracy
  complex<double> I = complex<double>(0.0, 1.0); // the imaginary unit

  // generate some random nonuniform points (x) and complex strengths (c)...
  vector<double> x(M);
  vector<complex<double>> c(M * ntrans);
  for (int j = 0; j < M; ++j)
    x[j] = PI * (2 * ((double)rand() / RAND_MAX) - 1); // uniform random in [-pi,pi)
  for (int j = 0; j < M * ntrans; ++j)                   // fill all ntrans vectors...
    c[j] =
        2 * ((double)rand() / RAND_MAX) - 1 + I * (2 * ((double)rand() / RAND_MAX) - 1);
  // allocate output array for the Fourier modes...
  vector<complex<double>> F(N * ntrans);

  // call the NUFFT (with iflag=+1): note pointers (not STL vecs) passed...
  int ier = finufft1d1many(ntrans, M, &x[0], &c[0], +1, tol, N, &F[0], NULL);
  // docs-end: many1d1
  if (ier) return ier; // no valid output to read

  int k   = 142519; // check the answer just for this mode, in every transform...
  assert(k >= -(double)N / 2 && k < (double)N / 2);

  double err = 0.0;
  for (int trans = 0; trans < ntrans; ++trans) {
    complex<double> Ftest = complex<double>(0, 0);           // do the naive calc...
    for (int j = 0; j < M; ++j)
      Ftest += c[j + M * trans] * exp(I * (double)k * x[j]); // c from transform # trans
    double Fmax = 0.0; // compute inf norm of F for transform # trans
    for (int m = 0; m < N; ++m) {
      double aF = abs(F[m + N * trans]);
      if (!isfinite(aF)) return 1; // any NaN/Inf fails
      if (aF > Fmax) Fmax = aF;
    }
    int kout    = k + N / 2 + N * trans; // output index, freq mode k, transform # trans
    double terr = abs(F[kout] - Ftest) / Fmax;
    if (!(terr <= 10 * tol)) return 1;   // also catches NaN
    if (terr > err) err = terr;
    printf("\ttransform %d, rel err in F[%d] is %.3g\n", trans, k, terr);
  }
  printf("1D type-1 double-prec NUFFT done. ier=%d, worst rel err over %d transforms "
         "in F[%d] is %.3g\n",
         ier, ntrans, k, err);
  return 0;
}
