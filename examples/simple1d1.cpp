// this is all you must include for the finufft lib...
#include <finufft.h>

// also used in this example...
#include <cassert>
#include <complex>
#include <cstdio>
#include <stdlib.h>
#include <vector>
using namespace std;

int main(int argc, char *argv[])
/* Example of calling the FINUFFT library from C++, using STL
   double complex vectors, with a math test.
   Double-precision version (see simple1d1f for single-precision)

   Compile with (static library case):
   g++ simple1d1.cpp -I../include ../lib-static/libfinufft.a -o simple1d1 -lfftw3
   -lfftw3_omp or if you have built a single-core version: g++ simple1d1.cpp -I../include
   ../lib-static/libfinufft.a -o simple1d1 -lfftw3

   Usage: ./simple1d1

   Also see ../docs/cex.rst or online documentation.
*/
{
  int M              = 1e6;                      // number of nonuniform points
  int N              = 1e6;                      // number of modes
  double acc         = 1e-9;                     // desired accuracy
  finufft_opts *opts = new finufft_opts;         // opts is pointer to struct
  finufft_default_opts(opts);
  complex<double> I = complex<double>(0.0, 1.0); // the imaginary unit

  // generate some random nonuniform points (x) and complex strengths (c)...
  vector<double> x(M);
  vector<complex<double>> c(M);
  for (int j = 0; j < M; ++j) {
    x[j] = M_PI * (2 * ((double)rand() / RAND_MAX) - 1); // uniform random in [-pi,pi)
    c[j] =
        2 * ((double)rand() / RAND_MAX) - 1 + I * (2 * ((double)rand() / RAND_MAX) - 1);
  }
  // allocate output array for the Fourier modes...
  vector<complex<double>> F(N);

  // call the NUFFT (with iflag=+1): note pointers (not STL vecs) passed...
  int ier = finufft1d1(M, &x[0], &c[0], +1, acc, N, &F[0], opts);

  int k = 142519; // check the answer just for this mode frequency...
  assert(k >= -(double)N / 2 && k < (double)N / 2);
  complex<double> Ftest = complex<double>(0, 0);
  for (int j = 0; j < M; ++j) Ftest += c[j] * exp(I * (double)k * x[j]);
  double Fmax = 0.0; // compute inf norm of F
  for (int m = 0; m < N; ++m) {
    double aF = abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }
  int kout   = k + N / 2; // index in output array for freq mode k
  double err = abs(F[kout] - Ftest) / Fmax;
  printf("1D type-1 double-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n", ier, k,
         err);
  return ier;
}
