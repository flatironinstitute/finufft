// this is all you must include for the finufft lib...
#include <complex>
#include <finufft.h>

// specific to this example...
#include <algorithm>
#include <cassert>
#include <math.h>
#include <random>
#include <stdio.h>
#include <vector>

// only good for small projects...
using namespace std;

static const double PI = 3.141592653589793238462643383279502884;
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
  int M      = 1e6;      // number of nonuniform points
  int N      = 1e6;      // number of modes
  double tol = 1e-9;     // desired accuracy

  int type = 1, dim = 1; // 1d1
  int64_t Ns[3];         // guru describes mode array by vector [N1,N2..]
  Ns[0]       = N;
  int ntransf = 1;       // we want to do a single transform at a time
  finufft_plan plan;     // creates a plan struct
  int ier;
  int changeopts = 0;    // do you want to try changing opts? 0 or 1
  if (changeopts) {      // demo how to change options away from defaults..
    finufft_opts opts;
    finufft_default_opts(&opts);
    opts.debug = 1; // example options change
    ier        = finufft_makeplan(type, dim, Ns, +1, ntransf, tol, &plan, &opts);
  } else            // or, nullptr here means use default opts...
    ier = finufft_makeplan(type, dim, Ns, +1, ntransf, tol, &plan, nullptr);
  if (ier) return ier;

  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-PI, PI), u1(-1.0, 1.0);

  // generate some random nonuniform points
  vector<double> x(M);
  generate(x.begin(), x.end(), [&] { return upi(rng); }); // uniform random in [-pi,pi)
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  ier = finufft_setpts(plan, M, x.data(), nullptr, nullptr, 0, nullptr, nullptr, nullptr);
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
  int n = 142519;                                   // check the answer just for this mode
  assert(n >= -double(N) / 2 && n < double(N) / 2); // ensure meaningful test
  complex<double> Ftest = complex<double>(0, 0);
  for (int j = 0; j < M; ++j) Ftest += c[j] * exp(1i * double(n) * x[j]);
  int nout    = n + N / 2; // index in output array for freq mode n
  double Fmax = 0.0;       // compute inf norm of F
  for (int m = 0; m < N; ++m) {
    double aF = abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }
  double err = abs(F[nout] - Ftest) / Fmax;
  printf("guru 1D type-1 double-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n", ier,
         n, err);
  if (!(err <= 10 * tol)) return 1;

  return ier;
}
