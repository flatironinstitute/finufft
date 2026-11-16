/* Example of calling the cuFINUFFT library from C++, using thrust device
   vectors, with a math test. Many 2D type-3 transforms with the same points,
   double precision.
   To compile, see README in this directory. Usage: ./example2d3many
*/
#include <cufinufft.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <random>
#include <thrust/device_vector.h>
#include <vector>
using namespace std;

constexpr double PI = 3.141592653589793238462643383279502884;

int main() {
  constexpr int M       = 1000000; // number of nonuniform points
  constexpr int N       = 500000;  // number of nonuniform target frequencies
  constexpr double S    = 64.0;    // target frequencies in [-S,S)^2
  constexpr int ntransf = 4;       // how many stacked transforms to do
  constexpr int iflag   = 1;       // sign of i in the transform math definition
  constexpr double tol  = 1e-6;
  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-PI, PI), u1(-1.0, 1.0), uS(-S, S);
  vector<double> x(M), y(M), s(N), t(N);
  vector<complex<double>> c(M * ntransf), fk(N * ntransf); // ntransf stacked vectors each
  generate(x.begin(), x.end(), [&] { return upi(rng); });  // points in [-pi,pi)^2
  generate(y.begin(), y.end(), [&] { return upi(rng); });
  generate(s.begin(), s.end(), [&] { return uS(rng); });   // target freqs in [-S,S)^2
  generate(t.begin(), t.end(), [&] { return uS(rng); });
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });

  thrust::device_vector<double> d_x = x, d_y = y, d_s = s, d_t = t; // copy to the GPU
  thrust::device_vector<complex<double>> d_c = c, d_fk = fk; // also c, and room for fk
  auto *dc  = reinterpret_cast<cuDoubleComplex *>(d_c.data().get()); // same layout
  auto *dfk = reinterpret_cast<cuDoubleComplex *>(d_fk.data().get());

  // guru (plan) interface, type 3
  cufinufft_plan plan;
  const int64_t nmodes[3] = {N, 1, 1};
  int ier = cufinufft_makeplan(3, 2, nmodes, iflag, ntransf, tol, &plan, nullptr);
  if (ier) return ier;
  ier = cufinufft_setpts(plan, M, d_x.data().get(), d_y.data().get(), nullptr, N,
                         d_s.data().get(), d_t.data().get(), nullptr);
  if (ier) return ier;
  ier = cufinufft_execute(plan, dc, dfk);
  if (ier) return ier;
  ier = cufinufft_destroy(plan);
  if (ier) return ier;
  thrust::copy(d_fk.begin(), d_fk.end(), fk.begin()); // copy fk back to the CPU

  const int tr = ntransf - 1;                         // check one transform, the last
  const int jt = N / 2;                               // check one target frequency
  const complex<double> I(0.0, 1.0);
  complex<double> fkt(0.0, 0.0);
  for (int j = 0; j < M; ++j)
    fkt += c[tr * M + j] * exp(I * (s[jt] * x[j] + t[jt] * y[j]));
  double Fmax = 0.0; // compute inf norm of fk
  for (const auto &Fm : fk) {
    const double a = abs(Fm);
    if (a > Fmax) Fmax = a;
  }
  const auto err = abs(fk[tr * N + jt] - fkt) / Fmax;
  if (!(err <= 10 * tol)) {
    fprintf(stderr, "FAILED: rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  printf("rel err %.3g\n", err);
  return 0;
}
