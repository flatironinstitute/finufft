// this is all you must include for the finufft lib...
#include <complex>
#include <finufft.h>

// specific to this example...
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

// only good for small projects...
using namespace std;

constexpr float pi = 3.14159265358979323846f;
// allows 1if to be the imaginary unit... (C++14 onwards)
using namespace std::complex_literals;

int main()
/* Example calling guru C++ interface to FINUFFT library, single-prec, passing
   pointers to STL vectors of C++ float complex numbers, with a math check.
   Barnett 7/5/20
   To compile, see README.  Usage: ./guru1d1f
*/
{
  constexpr int M     = 1e5;           // number of nonuniform points
  constexpr int N     = 1e4;           // number of modes
  constexpr float tol = 1e-3;          // desired accuracy

  constexpr int type = 1, dim = 1;     // 1d1
  constexpr int64_t Ns[3] = {N, 0, 0}; // guru describes mode array by vector [N1,N2..]
  constexpr int ntransf   = 1;         // we want to do a single transform at a time
  finufftf_plan plan;                  // creates single-prec plan struct: note the "f"
  int ier                  = 0;
  constexpr int changeopts = 1;        // do you want to try changing opts? 0 or 1
  if (changeopts) {                    // demo how to change options away from defaults..
    finufft_opts opts;
    finufftf_default_opts(&opts);      // note "f" for single-prec, throughout...
    opts.debug = 2;                    // example options change
    ier        = finufftf_makeplan(type, dim, Ns, +1, ntransf, tol, &plan, &opts);
  } else                               // or, NULL here means use default opts...
    ier = finufftf_makeplan(type, dim, Ns, +1, ntransf, tol, &plan, NULL);
  if (ier > 0) return ier;             // no plan to use; going on would segfault

  mt19937 rng(12345);
  uniform_real_distribution<float> upi(-pi, pi), u1(-1.0f, 1.0f);

  // generate some random nonuniform points
  vector<float> x(M);
  generate(x.begin(), x.end(), [&] { return upi(rng); });
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  ier = finufftf_setpts(plan, M, &x[0], NULL, NULL, 0, NULL, NULL, NULL);
  if (ier > 0) return ier; // the plan has no grid; executing it would segfault

  // generate some complex strengths
  vector<complex<float>> c(M);
  generate(c.begin(), c.end(), [&] { return complex<float>{u1(rng), u1(rng)}; });

  // alloc output array for the Fourier modes, then do the transform
  vector<complex<float>> F(N);
  ier = finufftf_execute(plan, &c[0], &F[0]);
  if (ier > 0) return ier;

  // for fun, do another with same NU pts (no re-sorting), but new strengths...
  generate(c.begin(), c.end(), [&] { return complex<float>{u1(rng), u1(rng)}; });
  ier = finufftf_execute(plan, &c[0], &F[0]);
  if (ier > 0) return ier;

  finufftf_destroy(plan); // done with transforms of this size

  // rest is math checking and reporting...
  constexpr int n = 1251; // check the answer just for this mode, must be in [-N/2,N/2)
  vector<complex<float>> terms(M);
  transform(c.begin(), c.end(), x.begin(), terms.begin(),
            [&](auto cj, auto xj) { return cj * exp(1if * float(n) * xj); });
  const auto Ftest = reduce(terms.begin(), terms.end());
  const auto Fmax = abs(
      *max_element(F.begin(), F.end(), [](auto a, auto b) { return abs(a) < abs(b); }));
  const int nout  = n + N / 2; // index in output array for freq mode n
  const auto err  = abs(F[nout] - Ftest) / Fmax;
  printf("guru 1D type-1 single-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n", ier,
         n, err);

  return !(err < 10 * tol);
}
