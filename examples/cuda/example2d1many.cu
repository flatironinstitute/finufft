/* Example of calling the cuFINUFFT library from C++, using thrust device
   vectors, with a math test. Many 2D type-1 transforms with the same points,
   single precision.
   To compile, see README in this directory. Usage: ./example2d1many
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

constexpr float PI = 3.141592653589793238462643383279502884f;

int main() {
  constexpr int M       = 65536; // number of nonuniform points
  constexpr int N1      = 256;   // numbers of output modes
  constexpr int N2      = 256;
  constexpr int ntransf = 2;     // how many stacked transforms to do
  constexpr int iflag   = 1;     // sign of i in the transform math definition
  constexpr float tol   = 1e-6f; // desired accuracy
  mt19937 rng(12345);
  uniform_real_distribution<float> upi(-PI, PI), u1(-1.0f, 1.0f);
  vector<float> x(M), y(M);
  vector<complex<float>> c(M * ntransf), fk(N1 * N2 * ntransf); // stacked vectors each
  generate(x.begin(), x.end(), [&] { return upi(rng); });       // points in [-pi,pi)^2
  generate(y.begin(), y.end(), [&] { return upi(rng); });
  generate(c.begin(), c.end(), [&] { return complex<float>{u1(rng), u1(rng)}; });

  thrust::device_vector<float> d_x = x, d_y = y;            // copy x,y to the GPU
  thrust::device_vector<complex<float>> d_c = c, d_fk = fk; // also c, and room for fk
  auto *dc  = reinterpret_cast<cuFloatComplex *>(d_c.data().get()); // same layout
  auto *dfk = reinterpret_cast<cuFloatComplex *>(d_fk.data().get());

  // guru (plan) interface, type 1
  cufinufftf_plan plan;
  const int64_t nmodes[3] = {N1, N2, 1};
  int ier = cufinufftf_makeplan(1, 2, nmodes, iflag, ntransf, tol, &plan, nullptr);
  if (ier) return ier;
  ier = cufinufftf_setpts(plan, M, d_x.data().get(), d_y.data().get(), nullptr, 0,
                          nullptr, nullptr, nullptr);
  if (ier) return ier;
  ier = cufinufftf_execute(plan, dc, dfk);
  if (ier) return ier;
  ier = cufinufftf_destroy(plan);
  if (ier) return ier;
  thrust::copy(d_fk.begin(), d_fk.end(), fk.begin()); // copy fk back to the CPU

  const int N   = N1 * N2;
  const int i   = ntransf - 1;                          // check one transform, the last
  const int nt1 = int(0.37 * N1), nt2 = int(0.26 * N2); // one mode index to check
  const complex<float> I(0.0f, 1.0f);
  complex<float> Ftest(0.0f, 0.0f);
  for (int j = 0; j < M; ++j)
    Ftest += c[i * M + j] * exp(I * (nt1 * x[j] + nt2 * y[j])); // c from transform # i
  float Fmax = 0.0f;                                            // compute inf norm of F
  for (const auto &Fm : fk) {
    const float a = abs(Fm);
    if (a > Fmax) Fmax = a;
  }
  const int it   = N1 / 2 + nt1 + N1 * (N2 / 2 + nt2); // index in F for this mode pair
  const auto err = abs(fk[it + i * N] - Ftest) / Fmax;
  if (!(err <= 10 * tol)) {
    fprintf(stderr, "FAILED: rel err %.3g > %.3g\n", double(err), double(10 * tol));
    return 1;
  }
  printf("rel err %.3g\n", double(err));
  return 0;
}
