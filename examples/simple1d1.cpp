// this is all you must include for the finufft lib...
#include <finufft.h>

// also used in this example...
#include <cassert>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const double PI = 3.141592653589793238462643383279502884;

int main(int argc, char *argv[])
/* Example of calling the FINUFFT library from C++, using STL
   double complex vectors, with a math test.
   Double-precision version (see simple1d1f for single-precision).
   To compile, see README in this directory.
   Also see ../docs/cex.rst or online documentation.
   Usage: ./simple1d1
*/
{
  int M              = 1e6;                      // number of nonuniform points
  int N              = 1e6;                      // number of modes
  double acc         = 1e-9;                     // desired accuracy
  finufft_opts *opts = new finufft_opts;         // opts is pointer to struct
  finufft_default_opts(opts);
  std::complex<double> I = std::complex<double>(0.0, 1.0); // the imaginary unit

  // generate some random nonuniform points (x) and complex strengths (c)...
  std::vector<double> x(M);
  std::vector<std::complex<double>> c(M);
  for (int j = 0; j < M; ++j) {
    x[j] = PI * (2 * ((double)std::rand() / RAND_MAX) - 1); // uniform random in [-pi,pi)
    c[j] =
        2 * ((double)std::rand() / RAND_MAX) - 1 + I * (2 * ((double)std::rand() / RAND_MAX) - 1);
  }
  // allocate output array for the Fourier modes...
  std::vector<std::complex<double>> F(N);

  // call the NUFFT (with iflag=+1): note pointers (not STL vecs) passed...
  int ier = finufft1d1(M, &x[0], &c[0], +1, acc, N, &F[0], opts);

  int k = 142519; // check the answer just for this mode frequency...
  assert(k >= -(double)N / 2 && k < (double)N / 2);
  std::complex<double> Ftest = std::complex<double>(0, 0);
  for (int j = 0; j < M; ++j) Ftest += c[j] * std::exp(I * (double)k * x[j]);
  double Fmax = 0.0; // compute inf norm of F
  for (int m = 0; m < N; ++m) {
    double aF = std::abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }
  int kout   = k + N / 2; // index in output array for freq mode k
  double err = std::abs(F[kout] - Ftest) / Fmax;
  printf("1D type-1 double-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n", ier, k,
         err);
  return ier;
}
