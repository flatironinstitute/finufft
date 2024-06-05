// this is all you must include for the finufft lib...
#include <complex>
#include <finufft.h>

// specific to this example...
#include <cassert>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// only good for small projects...
using namespace std;
// allows 1i to be the imaginary unit... (C++14 onwards)
using namespace std::complex_literals;

int main(int argc, char *argv[])
/* Example calling guru C++ interface to FINUFFT library, passing
   pointers to STL vectors of C++ double complex numbers, with a math check.
   Barnett 2/27/20

   Compile on linux with (or see ../makefile):
   g++ -std=c++14 -fopenmp guru1d1.cpp -I../include ../lib-static/libfinufft.a -o guru1d1
   -lfftw3 -lfftw3_omp -lm

   Or if you have built a single-thread library, remove -fopenmp and -lfftw3_omp

   Usage: ./guru1d1
*/
{
  int M      = 1e6;      // number of nonuniform points
  int N      = 1e6;      // number of modes
  double tol = 1e-9;     // desired accuracy

  int type = 1, dim = 1; // 1d1
  int64_t Ns[3];         // guru describes mode array by vector [N1,N2..]
  Ns[0]       = N;
  int ntransf = 1;       // we want to do a single transform at a time
  finufft_plan plan;     // creates a plan struct
  int changeopts = 0;    // do you want to try changing opts? 0 or 1
  if (changeopts) {      // demo how to change options away from defaults..
    finufft_opts opts;
    finufft_default_opts(&opts);
    opts.debug = 1; // example options change
    finufft_makeplan(type, dim, Ns, +1, ntransf, tol, &plan, &opts);
  } else            // or, NULL here means use default opts...
    finufft_makeplan(type, dim, Ns, +1, ntransf, tol, &plan, NULL);

  // generate some random nonuniform points
  vector<double> x(M);
  for (int j = 0; j < M; ++j)
    x[j] = M_PI * (2 * ((double)rand() / RAND_MAX) - 1); // uniform random in [-pi,pi)
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  finufft_setpts(plan, M, x.data(), NULL, NULL, 0, NULL, NULL, NULL);

  // generate some complex strengths
  vector<complex<double>> c(M);
  for (int j = 0; j < M; ++j)
    c[j] =
        2 * ((double)rand() / RAND_MAX) - 1 + 1i * (2 * ((double)rand() / RAND_MAX) - 1);

  // alloc output array for the Fourier modes, then do the transform
  vector<complex<double>> F(N);
  int ier = finufft_execute(plan, c.data(), F.data());

  // for fun, do another with same NU pts (no re-sorting), but new strengths...
  for (int j = 0; j < M; ++j)
    c[j] =
        2 * ((double)rand() / RAND_MAX) - 1 + 1i * (2 * ((double)rand() / RAND_MAX) - 1);
  ier = finufft_execute(plan, c.data(), F.data());

  finufft_destroy(plan); // don't forget! done with transforms of this size

  // rest is math checking and reporting...
  int n = 142519;                                   // check the answer just for this mode
  assert(n >= -(double)N / 2 && n < (double)N / 2); // ensure meaningful test
  complex<double> Ftest = complex<double>(0, 0);
  for (int j = 0; j < M; ++j) Ftest += c[j] * exp(1i * (double)n * x[j]);
  int nout    = n + N / 2; // index in output array for freq mode n
  double Fmax = 0.0;       // compute inf norm of F
  for (int m = 0; m < N; ++m) {
    double aF = abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }
  double err = abs(F[nout] - Ftest) / Fmax;
  printf("guru 1D type-1 double-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n", ier,
         n, err);

  return ier;
}
