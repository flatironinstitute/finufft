// this is all you must include...
#include <finufft.h>

// also needed for this example...
#include <cassert>
#include <complex>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
using namespace std;

int main(int argc, char *argv[])
/* Example of calling the FINUFFT library from C++, using STL
   single complex vectors, with a math test.
   (See simple1d1 for double-precision version.)

   Compile with:
   g++ -fopenmp simple1d1f.cpp -I../include ../lib-static/libfinufft.a -o simple1d1f
   -lfftw3f -lfftw3f_omp -lm or if you have built a single-core version: g++
   simple1d1f.cpp -I../include ../lib-static/libfinufft.a -o simple1d1f -lfftw3f -lm

   Usage: ./simple1d1f
*/
{
  int M              = 1e5;  // number of nonuniform points
  int N              = 1e5;  // number of modes (NB if too large lose acc in 1d)
  float acc          = 1e-3; // desired accuracy
  finufft_opts *opts = new finufft_opts;       // opts is pointer to struct
  finufftf_default_opts(opts);                 // note finufft "f" suffix
  complex<float> I = complex<float>(0.0, 1.0); // the imaginary unit

  // generate some random nonuniform points (x) and complex strengths (c)...
  vector<float> x(M);
  vector<complex<float>> c(M);
  for (int j = 0; j < M; ++j) {
    x[j] = M_PI * (2 * ((float)rand() / RAND_MAX) - 1); // uniform random in [-pi,pi)
    c[j] = 2 * ((float)rand() / RAND_MAX) - 1 + I * (2 * ((float)rand() / RAND_MAX) - 1);
  }
  // allocate output array for the Fourier modes...
  vector<complex<float>> F(N);

  // call the NUFFT (with iflag=+1): note pointers (not STL vecs) passed...
  int ier = finufftf1d1(M, &x[0], &c[0], +1, acc, N, &F[0], opts); // note "f"

  int k = 14251; // check the answer just for this mode...
  assert(k >= -(double)N / 2 && k < (double)N / 2);
  complex<float> Ftest = complex<float>(0, 0);
  for (int j = 0; j < M; ++j) Ftest += c[j] * exp(I * (float)k * x[j]);
  float Fmax = 0.0; // compute inf norm of F
  for (int m = 0; m < N; ++m) {
    float aF = abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }
  int kout  = k + N / 2; // index in output array for freq mode k
  float err = abs(F[kout] - Ftest) / Fmax;
  printf("1D type-1 single-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n", ier, k,
         err);
  return ier;
}
