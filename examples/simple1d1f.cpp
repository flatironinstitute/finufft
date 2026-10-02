// this is all you must include...
#include <finufft.h>

// also needed for this example...
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>
using namespace std;

constexpr float pi = 3.14159265358979323846f;

int main()
/* Example of calling the FINUFFT library from C++, using STL
   single complex vectors, with a math test.
   (See simple1d1 for double-precision version.)
   To compile, see README. Usage: ./simple1d1f
*/
{
  constexpr int M     = 1e5;                   // number of nonuniform points
  constexpr int N     = 1e4;                   // number of modes
  constexpr float acc = 1e-3;                  // desired accuracy
  finufft_opts opts;                           // opts is a plain struct
  finufftf_default_opts(&opts);                // note finufft "f" suffix
  complex<float> I = complex<float>(0.0, 1.0); // the imaginary unit

  // generate some random nonuniform points (x) and complex strengths (c)...
  mt19937 rng(12345);
  uniform_real_distribution<float> upi(-pi, pi), u1(-1.0f, 1.0f);
  vector<float> x(M);
  vector<complex<float>> c(M), F(N);
  generate(x.begin(), x.end(), [&] { return upi(rng); });
  generate(c.begin(), c.end(), [&] { return complex<float>{u1(rng), u1(rng)}; });
  // call the NUFFT (with iflag=+1): note pointers (not STL vecs) passed...
  int ier = finufftf1d1(M, &x[0], &c[0], +1, acc, N, &F[0], &opts); // note "f"
  if (ier) return ier;
  constexpr int k = 1425; // check the answer just for this mode...
  assert(k >= -(double)N / 2 && k < (double)N / 2);
  vector<complex<float>> terms(M);
  transform(c.begin(), c.end(), x.begin(), terms.begin(),
            [&](auto cj, auto xj) { return cj * exp(I * float(k) * xj); });
  const auto Ftest = reduce(terms.begin(), terms.end());
  const auto Fmax = abs(
      *max_element(F.begin(), F.end(), [](auto a, auto b) { return abs(a) < abs(b); }));
  const auto err = abs(F[k + N / 2] - Ftest) / Fmax; // k + N/2: index of freq mode k
  printf("1D type-1 single-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n", ier, k,
         err);
  return !(err < 10 * acc);
}
