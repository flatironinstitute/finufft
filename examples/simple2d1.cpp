#include <complex>
#include <finufft.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
using namespace std;

constexpr double pi = 3.14159265358979323846;

int main() {

  /* Simple 2D type-1 example of calling the FINUFFT library from C++, using plain
     arrays of C++ complex numbers, with a math test. Double precision version.
  */

  constexpr int M      = 1e6;  // number of nonuniform points
  constexpr int N      = 1e6;  // approximate total number of modes (N1*N2)
  constexpr double tol = 1e-6; // desired accuracy
  finufft_opts opts;
  finufft_default_opts(&opts);
  complex<double> I(0.0, 1.0); // the imaginary unit

  // generate random non-uniform points on (x,y) and complex strengths (c):
  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);
  vector<double> x(M), y(M);
  vector<complex<double>> c(M);
  // docs-start: simple2d1
  // each component of c uniform random in [-1,1]
  generate(x.begin(), x.end(), [&] { return upi(rng); });
  generate(y.begin(), y.end(), [&] { return upi(rng); });
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });

  // choose numbers of output Fourier coefficients in each dimension
  const int N1 = round(2.0 * sqrt(N));
  const int N2 = round(N / N1);

  // output array for the Fourier modes
  vector<complex<double>> F(N1 * N2);

  // call the NUFFT (with iflag += 1): note passing in pointers...
  int ier = finufft2d1(M, &x[0], &y[0], &c[0], 1, tol, N1, N2, &F[0], &opts);
  // docs-end: simple2d1
  if (ier) return ier;

  const int k1 = round(0.45 * N1); // check the answer for mode frequency (k1,k2)
  const int k2 = round(-0.35 * N2);

  complex<double> Ftest(0.0, 0.0);
  for (int j = 0; j < M; ++j)
    Ftest += c[j] * exp(I * (double(k1) * x[j] + double(k2) * y[j]));

  double Fmax = 0.0; // compute inf norm of F
  bool Ffin   = true;
  for (const auto &Fm : F) {
    const double a = abs(Fm);
    Ffin &= std::isfinite(a);
    if (a > Fmax) Fmax = a;
  }

  // indices in output array for this frequency pair (k1,k2)
  const int k1out    = k1 + N1 / 2;
  const int k2out    = k2 + N2 / 2;
  const int indexOut = k1out + k2out * (N1);

  // compute relative error
  const auto err     = abs(F[indexOut] - Ftest) / Fmax;
  if (!(Ffin && Fmax > 0.0 && err < 10 * tol)) {
    fprintf(stderr, "FAILED: F non-finite or rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  printf("rel err %.3g\n", err);
  return 0;
}
