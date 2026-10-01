// this is all you must include for the finufft lib...
#include <finufft.h>

// also used in this example...
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numeric>
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
   Expected output: multiple text lines (however many default threads), each
   reporting small error.
*/
{
  constexpr int M      = 1e5;                    // number of nonuniform points
  constexpr int N      = 1e5;                    // number of modes
  constexpr double acc = 1e-9;                   // desired accuracy
  finufft_opts opts;                             // opts is a plain struct
  finufft_default_opts(&opts);
  complex<double> I = complex<double>(0.0, 1.0); // the imaginary unit

  opts.nthreads = 1; // *crucial* so that each call single-thread (otherwise segfaults)
  int overallstatus = 0;

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
    int ier = finufft1d1(M, &x[0], &c[0], +1, acc, N, &F[0], &opts);
    if (ier > 0) overallstatus = 1;
    constexpr int k = 42519; // check the answer just for this mode frequency...
    assert(k >= -(double)N / 2 && k < (double)N / 2);
    vector<complex<double>> terms(M);
    transform(c.begin(), c.end(), x.begin(), terms.begin(),
              [&](auto cj, auto xj) { return cj * exp(I * double(k) * xj); });
    const auto Ftest = reduce(terms.begin(), terms.end());
    const auto Fmax = abs(
        *max_element(F.begin(), F.end(), [](auto a, auto b) { return abs(a) < abs(b); }));
    const auto err = abs(F[k + N / 2] - Ftest) / Fmax;   // k + N/2: index of freq mode k
    if (!(err < 10 * acc)) overallstatus = 1;            // also catches NaN

    printf("[thread %2d] 1D t-1 dbl-prec NUFFT done. ier=%d, rel err in F[%d]: %.3g\n",
           omp_get_thread_num(), ier, k, err);
  }

  return overallstatus;
}
