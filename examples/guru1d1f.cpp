// this is all you must include for the finufft lib...
#include <complex>
#include <finufft.h>

// specific to this example...
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// only good for small projects...
using namespace std;
// allows 1i to be the imaginary unit... (C++14 onwards)
using namespace std::complex_literals;

int main(int argc, char *argv[])
/* Example calling guru C++ interface to FINUFFT library, single-prec, passing
   pointers to STL vectors of C++ float complex numbers, with a math check.
   Barnett 7/5/20

   Compile on linux with:
   g++-7 -std=c++14 -fopenmp guru1d1f.cpp -I../include ../lib-static/libfinufft.a -o
   guru1d1f  -lfftw3f -lfftw3f_omp -lm

   Or if you have built a single-core library, remove -fopenmp and -lfftw3f_omp

   Usage: ./guru1d1f
*/
{
  int M     = 1e5;                // number of nonuniform points
  int N     = 1e5;                // number of modes
  float tol = 1e-5;               // desired accuracy

  int type = 1, dim = 1;          // 1d1
  int64_t Ns[3];                  // guru describes mode array by vector [N1,N2..]
  Ns[0]       = N;
  int ntransf = 1;                // we want to do a single transform at a time
  finufftf_plan plan;             // creates single-prec plan struct: note the "f"
  int changeopts = 1;             // do you want to try changing opts? 0 or 1
  if (changeopts) {               // demo how to change options away from defaults..
    finufft_opts opts;
    finufftf_default_opts(&opts); // note "f" for single-prec, throughout...
    opts.debug = 2;               // example options change
    finufftf_makeplan(type, dim, Ns, +1, ntransf, tol, &plan, &opts);
  } else                          // or, NULL here means use default opts...
    finufftf_makeplan(type, dim, Ns, +1, ntransf, tol, &plan, NULL);

  // generate some random nonuniform points
  vector<float> x(M);
  for (int j = 0; j < M; ++j)
    x[j] = M_PI * (2 * ((float)rand() / RAND_MAX) - 1); // uniform random in [-pi,pi)
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  finufftf_setpts(plan, M, &x[0], NULL, NULL, 0, NULL, NULL, NULL);

  // generate some complex strengths
  vector<complex<float>> c(M);
  for (int j = 0; j < M; ++j)
    c[j] =
        2 * ((float)rand() / RAND_MAX) - 1 + 1if * (2 * ((float)rand() / RAND_MAX) - 1);

  // alloc output array for the Fourier modes, then do the transform
  vector<complex<float>> F(N);
  int ier = finufftf_execute(plan, &c[0], &F[0]);

  // for fun, do another with same NU pts (no re-sorting), but new strengths...
  for (int j = 0; j < M; ++j)
    c[j] =
        2 * ((float)rand() / RAND_MAX) - 1 + 1if * (2 * ((float)rand() / RAND_MAX) - 1);
  ier = finufftf_execute(plan, &c[0], &F[0]);

  finufftf_destroy(plan); // done with transforms of this size

  // rest is math checking and reporting...
  int n = 12519; // check the answer just for this mode, must be in [-N/2,N/2)
  complex<float> Ftest = complex<float>(0, 0);
  for (int j = 0; j < M; ++j) Ftest += c[j] * exp(1if * (float)n * x[j]);
  int nout   = n + N / 2; // index in output array for freq mode n
  float Fmax = 0.0;       // compute inf norm of F
  for (int m = 0; m < N; ++m) {
    float aF = abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }
  float err = abs(F[nout] - Ftest) / Fmax;
  printf("guru 1D type-1 single-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n", ier,
         n, err);

  return ier;
}
