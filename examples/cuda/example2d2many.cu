/* Example of calling the cuFINUFFT library from C++, using thrust device
   vectors, with a math test. Many 2D type-2 transforms with the same points,
   double precision.
   To compile, see README in this directory. Usage: ./example2d2many
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
  constexpr int N1           = 128;     // numbers of input modes
  constexpr int N2           = 128;
  constexpr int M            = 1000000; // number of nonuniform points
  constexpr int ntransf      = 4;       // how many stacked transforms to do
  constexpr int maxbatchsize = 4;       // max batch size in the internal transforms
  constexpr int iflag        = 1;       // sign of i in the transform math definition
  constexpr double tol       = 1e-6;
  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-PI, PI), u1(-1.0, 1.0);
  vector<double> x(M), y(M);
  vector<complex<double>> c(M * ntransf), fk(N1 * N2 * ntransf); // stacked vectors each
  generate(x.begin(), x.end(), [&] { return upi(rng); });        // points in [-pi,pi)^2
  generate(y.begin(), y.end(), [&] { return upi(rng); });
  generate(fk.begin(), fk.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });

  thrust::device_vector<double> d_x = x, d_y = y;            // copy x,y to the GPU
  thrust::device_vector<complex<double>> d_fk = fk, d_c = c; // also fk, and room for c
  auto *dfk = reinterpret_cast<cuDoubleComplex *>(d_fk.data().get()); // same layout
  auto *dc  = reinterpret_cast<cuDoubleComplex *>(d_c.data().get());

  // guru (plan) interface, type 2
  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.gpu_maxbatchsize = maxbatchsize;
  cufinufft_plan plan;
  const int64_t nmodes[3] = {N1, N2, 1};
  int ier = cufinufft_makeplan(2, 2, nmodes, iflag, ntransf, tol, &plan, &opts);
  if (ier) return ier;
  ier = cufinufft_setpts(plan, M, d_x.data().get(), d_y.data().get(), nullptr, 0, nullptr,
                         nullptr, nullptr);
  if (ier) return ier;
  ier = cufinufft_execute(plan, dc, dfk);
  if (ier) return ier;
  ier = cufinufft_destroy(plan);
  if (ier) return ier;
  thrust::copy(d_c.begin(), d_c.end(), c.begin()); // copy c back to the CPU

  const int t  = ntransf - 1;                      // check one transform, the last
  const int jt = M / 2;                            // check one target point
  const complex<double> I(0.0, 1.0);
  complex<double> ct(0.0, 0.0);
  int m = 0;
  for (int m2 = -(N2 / 2); m2 <= (N2 - 1) / 2; ++m2) // loop in correct order over F
    for (int m1 = -(N1 / 2); m1 <= (N1 - 1) / 2; ++m1)
      ct += fk[t * N1 * N2 + m++] * exp(I * (m1 * x[jt] + m2 * y[jt])); // direct sum
  double cmax = 0.0; // compute inf norm of c
  for (const auto &cm : c) {
    const double a = abs(cm);
    if (a > cmax) cmax = a;
  }
  const auto err = abs(c[t * M + jt] - ct) / cmax;
  if (!(err <= 10 * tol)) {
    fprintf(stderr, "FAILED: rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  printf("rel err %.3g\n", err);
  return 0;
}
