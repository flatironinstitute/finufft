/* Example of calling the vectorized cuFINUFFT library from C++, using thrust
   device vectors, with a math test. GPU version of examples/many1d1.cpp.
   Double-precision version.
   To compile, see README in this directory. Usage: ./many1d1
*/
#include <cufinufft.h>

#include <algorithm>
#include <complex>
#include <cstdio>
#include <random>
#include <thrust/device_vector.h>
#include <vector>
using namespace std;

constexpr double PI = 3.141592653589793238462643383279502884;

int main() {
  constexpr int ntrans = 3;    // how many stacked transforms to do
  constexpr int M      = 2e6;  // number of nonuniform points
  constexpr int N      = 1e6;  // number of output modes
  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-PI, PI), u1(-1.0, 1.0);
  vector<double> x(M);
  vector<complex<double>> c(M * ntrans), F(N * ntrans);   // ntrans stacked vectors each
  generate(x.begin(), x.end(), [&] { return upi(rng); }); // points in [-pi,pi)
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });

  thrust::device_vector<double> d_x          = x;          // copy x to the GPU
  thrust::device_vector<complex<double>> d_c = c, d_f = F; // also c, and room for F
  auto *dc = reinterpret_cast<cuDoubleComplex *>(d_c.data().get()); // same layout
  auto *df = reinterpret_cast<cuDoubleComplex *>(d_f.data().get());
  // nullptr: default opts
  int ier  = cufinufft1d1many(ntrans, M, d_x.data().get(), dc, +1, 1e-9, N, df, nullptr);
  if (ier) return ier;
  thrust::copy(d_f.begin(), d_f.end(), F.begin()); // copy F back to the CPU

  constexpr double tol = 1e-9;
  const complex<double> I(0.0, 1.0);
  constexpr int k = 142519;     // check the answer just for this mode, in [-N/2,N/2)
  const int trans = ntrans - 1; // ...in this transform
  complex<double> Ftest(0.0, 0.0);
  for (int j = 0; j < M; ++j)
    Ftest += c[j + M * trans] * exp(I * double(k) * x[j]); // c from transform # trans
  double Fmax = 0.0;                                       // compute inf norm of F
  for (const auto &Fm : F) {
    const double a = abs(Fm);
    if (a > Fmax) Fmax = a;
  }
  const auto err = abs(F[k + N / 2 + N * trans] - Ftest) / Fmax; // output index of k
  if (!(err <= 10 * tol)) {
    fprintf(stderr, "FAILED: rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  printf("rel err %.3g\n", err);
  return 0;
}
