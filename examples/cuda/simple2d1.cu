/* Example of calling the cuFINUFFT library from C++, using thrust device
   vectors, with a math test. GPU version of examples/simple2d1.cpp.
   Simple 2D type-1 transform, double-precision version.
   To compile, see README in this directory. Usage: ./simple2d1
*/
#include <cufinufft.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <thrust/device_vector.h>
#include <vector>
using namespace std;

constexpr double PI = 3.141592653589793238462643383279502884;

int main() {
  constexpr int M = 2e6; // number of nonuniform points
  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-PI, PI), u1(-1.0, 1.0);
  vector<double> x(M), y(M);
  vector<complex<double>> c(M);
  generate(x.begin(), x.end(), [&] { return upi(rng); }); // points in [-pi,pi)^2
  generate(y.begin(), y.end(), [&] { return upi(rng); });
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });

  int N1 = 1000, N2 = 2000;
  vector<complex<double>> F(N1 * N2);
  thrust::device_vector<double> d_x = x, d_y = y;          // copy x,y to the GPU
  thrust::device_vector<complex<double>> d_c = c, d_f = F; // also c, and room for F
  auto *dc = reinterpret_cast<cuDoubleComplex *>(d_c.data().get()); // same layout
  auto *df = reinterpret_cast<cuDoubleComplex *>(d_f.data().get());
  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.upsampfac = 1.25;
  int ier = cufinufft2d1(M, d_x.data().get(), d_y.data().get(), dc, +1, 1e-6, N1, N2, df,
                         &opts);
  if (ier) return ier;
  thrust::copy(d_f.begin(), d_f.end(), F.begin()); // copy F back to the CPU

  constexpr double tol = 1e-6;
  const complex<double> I(0.0, 1.0);
  const int k1 = round(0.45 * N1); // check the answer for mode frequency (k1,k2)
  const int k2 = round(-0.35 * N2);
  complex<double> Ftest(0.0, 0.0);
  for (int j = 0; j < M; ++j)
    Ftest += c[j] * exp(I * (double(k1) * x[j] + double(k2) * y[j]));
  double Fmax = 0.0; // compute inf norm of F
  for (const auto &Fm : F) {
    const double a = abs(Fm);
    if (a > Fmax) Fmax = a;
  }
  // index in output array for this frequency pair (k1,k2)
  const auto err = abs(F[k1 + N1 / 2 + (k2 + N2 / 2) * N1] - Ftest) / Fmax;
  if (!(err <= 10 * tol)) {
    fprintf(stderr, "FAILED: rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  printf("rel err %.3g\n", err);
  return 0;
}
