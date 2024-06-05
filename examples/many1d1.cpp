#include <finufft.h>

#include <cassert>
#include <complex>
#include <cstdio>
#include <stdlib.h>
#include <vector>
using namespace std;

int main(int argc, char *argv[])
/* Example of calling the vectorized FINUFFT library from C++, using STL
   double complex vectors, with a math test.

   Compile with:
   g++ -fopenmp many1d1.cpp -I../include ../lib-static/libfinufft.a -o many1d1 -lfftw3
   -lfftw3_omp -lm or if you have built a single-core version: g++ many1d1.cpp
   -I../include ../lib-static/libfinufft.a -o many1d1 -lfftw3 -lm

   Usage: ./many1d1
*/
{
  int ntrans         = 3;                // how many stacked transforms to do
  int M              = 1e6;              // nonuniform points (same for all transforms)
  int N              = 1e6;              // number of modes (same for all transforms)
  double tol         = 1e-9;             // desired accuracy
  finufft_opts *opts = new finufft_opts; // opts is pointer to struct
  finufft_default_opts(opts);
  complex<double> I = complex<double>(0.0, 1.0); // the imaginary unit

  // generate some random nonuniform points (x) and complex strengths (c)...
  vector<double> x(M);
  vector<complex<double>> c(M * ntrans);
  for (int j = 0; j < M; ++j)
    x[j] = M_PI * (2 * ((double)rand() / RAND_MAX) - 1); // uniform random in [-pi,pi)
  for (int j = 0; j < M * ntrans; ++j)                   // fill all ntrans vectors...
    c[j] =
        2 * ((double)rand() / RAND_MAX) - 1 + I * (2 * ((double)rand() / RAND_MAX) - 1);
  // allocate output array for the Fourier modes...
  vector<complex<double>> F(N * ntrans);

  // call the NUFFT (with iflag=+1): note pointers (not STL vecs) passed...
  int ier = finufft1d1many(ntrans, M, &x[0], &c[0], +1, tol, N, &F[0], NULL);

  int k     = 142519;     // check the answer just for this mode...
  int trans = ntrans - 1; // ...in this transform
  assert(k >= -(double)N / 2 && k < (double)N / 2);

  complex<double> Ftest = complex<double>(0, 0);           // do the naive calc...
  for (int j = 0; j < M; ++j)
    Ftest += c[j + M * trans] * exp(I * (double)k * x[j]); // c from transform # trans
  double Fmax = 0.0; // compute inf norm of F for transform # trans
  for (int m = 0; m < N; ++m) {
    double aF = abs(F[m + N * trans]);
    if (aF > Fmax) Fmax = aF;
  }
  int kout   = k + N / 2 + N * trans; // output index, freq mode k, transform # trans
  double err = abs(F[kout] - Ftest) / Fmax;
  printf("1D type-1 double-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n", ier, k,
         err);
  return ier;
}
