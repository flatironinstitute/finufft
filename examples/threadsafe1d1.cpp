// this is all you must include for the finufft lib...
#include <finufft.h>

// also used in this example...
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <omp.h>
#include <random>
#include <vector>
using namespace std;

constexpr double pi = 3.14159265358979323846;

int main()
/* Demo single-threaded FINUFFT calls from inside a OMP parallel block.
   Adapted from simple1d1.cpp: C++, STL double complex vectors, with math test.
   Barnett 4/19/21, eg for Goran Zauhar, issue #183. Also see: many1d1.cpp.
   To compile, see README.
   Usage: ./threadsafe1d1
   Exit code: 0 if all threads passed their math test, nonzero otherwise.
*/
{
  constexpr int M      = 1e5;                    // number of nonuniform points
  constexpr int N      = 1e5;                    // number of modes
  constexpr double tol = 1e-9;                   // desired accuracy
  double maxerr        = 0.0;                    // worst rel err over threads
  finufft_opts opts;                             // opts is a plain struct
  finufft_default_opts(&opts);
  complex<double> I = complex<double>(0.0, 1.0); // the imaginary unit

  opts.nthreads = 1; // *crucial* so that each call single-thread (otherwise segfaults)

  // Now have each thread do independent 1D type 1 on their own data:
#pragma omp parallel
  {
    // generate some random nonuniform points (x) and complex strengths (c)...
    // Note that these are local to the thread (if you have the *same* sets of
    // NU pts x for each thread, consider instead using one vectorized multithreaded
    // transform, which would be faster).
    mt19937 rng(12345 + omp_get_thread_num());
    uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);
    vector<double> x(M);
    vector<complex<double>> c(M), F(N); // F: output modes, local to the thread
    generate(x.begin(), x.end(), [&] { return upi(rng); });
    generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });
    // call the NUFFT (with iflag=+1): note pointers (not STL vecs) passed...
    int ier         = finufft1d1(M, &x[0], &c[0], +1, tol, N, &F[0], &opts);
    constexpr int k = 42519; // check the answer just for this mode frequency...
    complex<double> Ftest(0.0, 0.0);
    double Fmax = 0.0; // compute inf norm of F
    bool Ffin   = true;
    double err  = HUGE_VAL; // default to fail
    if (!ier) {
      for (int j = 0; j < M; ++j) Ftest += c[j] * exp(I * double(k) * x[j]);
      for (const auto &Fm : F) {
        const double a = abs(Fm);
        Ffin &= std::isfinite(a);
        if (a > Fmax) Fmax = a;
      }
      if (Ffin && Fmax > 0.0) err = abs(F[k + N / 2] - Ftest) / Fmax;
    }
#pragma omp critical
    {
      maxerr = err > maxerr ? err : maxerr; // track worst rel err across threads
      if (ier) maxerr = HUGE_VAL;
    }
  }

  if (!(maxerr < 10 * tol)) {
    fprintf(stderr, "FAILED: ier or F non-finite or rel err %.3g > %.3g\n", maxerr,
            10 * tol);
    return 1;
  }
  printf("rel err %.3g\n", maxerr);
  return 0;
}
