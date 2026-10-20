// this is all you must include for the finufft lib...
#include <complex>
#include <finufft.h>

// specific to this example...
#include <algorithm>
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
/* Example calling guru C++ interface to FINUFFT library, single-prec, passing
   pointers to STL vectors of C++ float complex numbers, with a math check.
   Barnett 7/5/20
   To compile, see README.  Usage: ./guru1d1f
*/
{
  int M     = 1e5;                // number of nonuniform points
  int N     = 1e4;                // number of modes
  float tol = 1e-3;               // desired accuracy

  int type = 1, dim = 1;          // 1d1
  int64_t Ns[3];                  // guru describes mode array by vector [N1,N2..]
  Ns[0]       = N;
  int ntransf = 1;                // we want to do a single transform at a time
  finufftf_plan plan;             // creates single-prec plan struct: note the "f"
  int ier        = 0;
  int changeopts = 1;             // do you want to try changing opts? 0 or 1
  if (changeopts) {               // demo how to change options away from defaults..
    finufft_opts opts;
    finufftf_default_opts(&opts); // note "f" for single-prec, throughout...
    opts.debug = 2;               // example options change
    ier        = finufftf_makeplan(type, dim, Ns, +1, ntransf, tol, &plan, &opts);
  } else                          // or, nullptr here means use default opts...
    ier = finufftf_makeplan(type, dim, Ns, +1, ntransf, tol, &plan, nullptr);
  if (ier > 0) return ier;        // no plan to use; going on would segfault

  mt19937 rng(12345);
  uniform_real_distribution<float> upi(-PI, PI), u1(-1.0f, 1.0f);

  // generate some random nonuniform points
  vector<float> x(M);
  generate(x.begin(), x.end(), [&] { return upi(rng); }); // uniform random in [-pi,pi)
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  ier =
      finufftf_setpts(plan, M, x.data(), nullptr, nullptr, 0, nullptr, nullptr, nullptr);
  if (ier > 0) return ier; // the plan has no grid; executing it would segfault

  // generate some complex strengths
  vector<complex<float>> c(M);
  generate(c.begin(), c.end(), [&] { return complex<float>{u1(rng), u1(rng)}; });

  // alloc output array for the Fourier modes, then do the transform
  vector<complex<float>> F(N);
  ier = finufftf_execute(plan, c.data(), F.data());
  if (ier > 0) return ier;

  // for fun, do another with same NU pts (no re-sorting), but new strengths...
  generate(c.begin(), c.end(), [&] { return complex<float>{u1(rng), u1(rng)}; });
  ier = finufftf_execute(plan, c.data(), F.data());
  if (ier > 0) return ier;

  finufftf_destroy(plan); // done with transforms of this size

  // rest is math checking and reporting...
  int n = 1251; // check the answer just for this mode, must be in [-N/2,N/2)
  complex<float> Ftest = complex<float>(0, 0);
  for (int j = 0; j < M; ++j) Ftest += c[j] * exp(1if * float(n) * x[j]);
  int nout   = n + N / 2; // index in output array for freq mode n
  float Fmax = 0.0;       // compute inf norm of F
  for (int m = 0; m < N; ++m) {
    float aF = abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }
  float err = abs(F[nout] - Ftest) / Fmax;
  printf("guru 1D type-1 single-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n", ier,
         n, err);
  if (!(err <= 10 * tol)) return 1;

  return ier;
}
