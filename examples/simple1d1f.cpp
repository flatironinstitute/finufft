// this is all you must include...
#include <finufft.h>

// also needed for this example...
#include <algorithm>
#include <cassert>
#include <complex>
#include <random>
#include <stdio.h>
#include <vector>
using namespace std;

static const double PI = 3.141592653589793238462643383279502884;

int main()
/* Example of calling the FINUFFT library from C++, using STL
   single complex vectors, with a math test.
   (See simple1d1 for double-precision version.)
   To compile, see README. Usage: ./simple1d1f
*/
{
  int M     = 1e5;                             // number of nonuniform points
  int N     = 1e4;                             // number of modes
  float acc = 1e-3;                            // desired accuracy
  finufft_opts opts;                           // opts is a plain struct
  finufftf_default_opts(&opts);                // note finufft "f" suffix
  complex<float> I = complex<float>(0.0, 1.0); // the imaginary unit

  // generate some random nonuniform points (x) and complex strengths (c)...
  mt19937 rng(12345);
  uniform_real_distribution<float> upi(-PI, PI), u1(-1.0f, 1.0f);
  vector<float> x(M);
  vector<complex<float>> c(M);
  generate(x.begin(), x.end(), [&] { return upi(rng); }); // uniform random in [-pi,pi)
  generate(c.begin(), c.end(), [&] { return complex<float>{u1(rng), u1(rng)}; });
  // allocate output array for the Fourier modes...
  vector<complex<float>> F(N);

  // call the NUFFT (with iflag=+1): note pointers (not STL vecs) passed...
  int ier = finufftf1d1(M, x.data(), c.data(), +1, acc, N, F.data(), &opts); // note "f"

  int k   = 1425; // check the answer just for this mode...
  assert(k >= -double(N) / 2 && k < double(N) / 2);
  complex<float> Ftest = complex<float>(0, 0);
  for (int j = 0; j < M; ++j) Ftest += c[j] * exp(I * float(k) * x[j]);
  float Fmax = 0.0; // compute inf norm of F
  for (int m = 0; m < N; ++m) {
    float aF = abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }
  int kout  = k + N / 2; // index in output array for freq mode k
  float err = abs(F[kout] - Ftest) / Fmax;
  printf("1D type-1 single-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n", ier, k,
         err);
  if (!(err <= 10 * acc)) return 1;
  return ier;
}
