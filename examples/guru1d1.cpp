// this is all you must include for the finufft lib...
#include <complex>
#include <finufft.h>

// specific to this example...
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

// only good for small projects...
using namespace std;

constexpr double pi = 3.14159265358979323846;
// allows 1i to be the imaginary unit... (C++14 onwards)
using namespace std::complex_literals;

int main()
/* Example calling guru C++ interface to FINUFFT library, passing
   pointers to STL vectors of C++ double complex numbers, with a math check.
   Barnett 2/27/20
   To compile see README. Also see ../docs/cex.rst
   Usage: ./guru1d1
*/
{
  constexpr int M      = 1e6;          // number of nonuniform points
  constexpr int N      = 1e6;          // number of modes
  constexpr double tol = 1e-9;         // desired accuracy

  constexpr int type = 1, dim = 1;     // 1d1
  constexpr int64_t Ns[3] = {N, 0, 0}; // guru describes mode array by vector [N1,N2..]
  constexpr int ntransf   = 1;         // we want to do a single transform at a time
  finufft_plan plan;                   // creates a plan struct
  constexpr int changeopts = 0;        // do you want to try changing opts? 0 or 1
  int ier;
  if (changeopts) {                    // demo how to change options away from defaults..
    finufft_opts opts;
    finufft_default_opts(&opts);
    opts.debug = 1; // example options change
    ier        = finufft_makeplan(type, dim, Ns, +1, ntransf, tol, &plan, &opts);
  } else            // or, NULL here means use default opts...
    ier = finufft_makeplan(type, dim, Ns, +1, ntransf, tol, &plan, NULL);
  if (ier) return ier;

  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);

  // generate some random nonuniform points
  vector<double> x(M);
  generate(x.begin(), x.end(), [&] { return upi(rng); });
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  ier = finufft_setpts(plan, M, x.data(), NULL, NULL, 0, NULL, NULL, NULL);
  if (ier) return ier;

  // generate some complex strengths
  vector<complex<double>> c(M);
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });

  // alloc output array for the Fourier modes, then do the transform
  vector<complex<double>> F(N);
  ier = finufft_execute(plan, c.data(), F.data());
  if (ier) return ier;

  // for fun, do another with same NU pts (no re-sorting), but new strengths...
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });
  ier = finufft_execute(plan, c.data(), F.data());
  if (ier) return ier;

  finufft_destroy(plan); // don't forget! done with transforms of this size

  // rest is math checking and reporting...
  constexpr int n = 142519;                         // check the answer just for this mode
  assert(n >= -(double)N / 2 && n < (double)N / 2); // ensure meaningful test
  vector<complex<double>> terms(M);
  transform(c.begin(), c.end(), x.begin(), terms.begin(),
            [&](auto cj, auto xj) { return cj * exp(1i * double(n) * xj); });
  const auto Ftest = reduce(terms.begin(), terms.end());
  const auto Fmax = abs(
      *max_element(F.begin(), F.end(), [](auto a, auto b) { return abs(a) < abs(b); }));
  const int nout   = n + N / 2; // index in output array for freq mode n
  const auto err   = abs(F[nout] - Ftest) / Fmax;
  printf("guru 1D type-1 double-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n", ier,
         n, err);

  return !(err < 10 * tol);
}
