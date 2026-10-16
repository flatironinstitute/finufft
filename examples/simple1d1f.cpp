// this is all you must include...
#include <finufft.h>

// also needed for this example...
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
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
  constexpr float tol = 1e-3;                  // desired accuracy
  finufft_opts opts;                           // opts is a plain struct
  finufftf_default_opts(&opts);                // note finufft "f" suffix
  complex<float> I = complex<float>(0.0, 1.0); // the imaginary unit

  // generate some random nonuniform points (x) and complex strengths (c)...
  mt19937 rng(12345);
  uniform_real_distribution<float> upi(-pi, pi), u1(-1.0f, 1.0f);
  vector<float> x(M);
  vector<complex<float>> c(M);
  generate(x.begin(), x.end(), [&] { return upi(rng); });
  generate(c.begin(), c.end(), [&] { return complex<float>{u1(rng), u1(rng)}; });
  // allocate output array for the Fourier modes...
  vector<complex<float>> F(N);
  // call the NUFFT (with iflag=+1): note pointers (not STL vecs) passed...
  int ier = finufftf1d1(M, &x[0], &c[0], +1, tol, N, &F[0], &opts); // note "f"
  if (ier) return ier;
  constexpr int k = 1425; // check the answer just for this mode...
  static_assert(k >= -(double)N / 2 && k < (double)N / 2); // ensure meaningful test
  complex<float> Ftest(0.0f, 0.0f);
  for (int j = 0; j < M; ++j) Ftest += c[j] * exp(I * float(k) * x[j]);
  float Fmax = 0.0f; // compute inf norm of F
  bool Ffin  = true;
  for (const auto &Fm : F) {
    const float a = abs(Fm);
    Ffin &= std::isfinite(a);
    if (a > Fmax) Fmax = a;
  }
  const auto err = abs(F[k + N / 2] - Ftest) / Fmax; // k + N/2: index of freq mode k
  if (!(Ffin && Fmax > 0.0f && err < 10 * tol)) {
    fprintf(stderr, "FAILED: F non-finite or rel err %.3g > %.3g\n", double(err),
            double(10 * tol));
    return 1;
  }
  printf("rel err %.3g\n", double(err));
  return 0;
}
