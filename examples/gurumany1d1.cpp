/* Demonstrate guru FINUFFT interface performing a stack of 1d type 1
   transforms in a single execute call. See guru1d1.cpp for other guru
   features demonstrated. Barnett 11/22/23
   To compile, see README.
   Usage: ./gurumany1d1           (exit code 0 indicates success)
*/

// this is all you must include for the finufft lib...
#include <complex>
#include <finufft.h>

// specific to this demo...
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

int main() {
  constexpr int M      = 2e5;          // number of nonuniform points
  constexpr int N      = 1e5;          // number of modes
  constexpr double tol = 1e-9;         // desired accuracy
  constexpr int ntrans = 100;          // request a bunch of transforms in the execute
  constexpr int isign  = +1;           // sign of i in the transform math definition

  constexpr int type = 1, dim = 1;     // 1d1
  constexpr int64_t Ns[3] = {N, 0, 0}; // guru describes mode array by vector [N1,N2..]
  finufft_plan plan;                   // creates a plan struct (NULL below: default opts)
  int ier = finufft_makeplan(type, dim, Ns, isign, ntrans, tol, &plan, NULL);
  if (ier) return ier;

  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);

  // generate random nonuniform points and pass to FINUFFT
  vector<double> x(M);
  generate(x.begin(), x.end(), [&] { return upi(rng); });
  ier = finufft_setpts(plan, M, x.data(), NULL, NULL, 0, NULL, NULL, NULL);
  if (ier) return ier;

  // generate ntrans complex strength vectors each of length M (the slow bit!)
  vector<complex<double>> c(M * ntrans); // plain contiguous storage
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });

  // alloc output array for the Fourier modes, then do the transform
  vector<complex<double>> F(N * ntrans);
  printf("guru many 1D type-1 double-prec, tol=%.3g, executing %d transforms "
         "(vectorized), each size %d NU pts to %d modes...\n",
         tol, ntrans, M, N);
  ier = finufft_execute(plan, c.data(), F.data());
  if (ier) return ier;

  // could now change c, do another execute, do another setpts, execute, etc...

  finufft_destroy(plan); // don't forget! we're done with transforms of this size

  // rest is math checking and reporting...
  constexpr int k     = 42519;                      // check the answer just for this mode
  constexpr int trans = ntrans - 1;                 // ...testing in just this transform
  assert(k >= -(double)N / 2 && k < (double)N / 2); // ensure meaningful test
  assert(trans >= 0 && trans < ntrans);
  vector<complex<double>> terms(M);
  transform(x.begin(), x.end(), c.begin() + M * trans, terms.begin(),
            [&](auto xj, auto cj) { return cj * exp(1i * double(k) * xj); }); // c offset
                                                                              // to trans
  const auto Ftest = reduce(terms.begin(), terms.end());
  const auto Ft = F.begin() + N * trans; // output of the selected transform
  const auto Fmax =
      abs(*max_element(Ft, Ft + N, [](auto a, auto b) { return abs(a) < abs(b); }));
  const int nout   = k + N / 2 + N * trans; // output index for freq mode k in the trans
  const auto err   = abs(F[nout] - Ftest) / Fmax;
  printf("\tdone: ier=%d; for transform %d, rel err in F[%d] is %.3g\n", ier, trans, k,
         err);

  return !(err < 10 * tol);
}
