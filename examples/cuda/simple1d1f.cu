/* Example of calling the cuFINUFFT library from C++, using thrust device
   vectors, with a math test. GPU version of examples/simple1d1f.cpp.
   Single-precision version (see simple1d1.cu for double-precision).
   To compile, see README in this directory. Usage: ./simple1d1f
*/
#include <cufinufft.h>

#include <algorithm>
#include <complex>
#include <cstdio>
#include <random>
#include <thrust/device_vector.h>
#include <vector>
using namespace std;

constexpr float PI = 3.141592653589793238462643383279502884f;

int main() {
  constexpr int M     = 1e5;  // number of nonuniform points
  constexpr int N     = 1e4;  // number of output modes
  constexpr float tol = 1e-3; // desired accuracy
  mt19937 rng(12345);
  uniform_real_distribution<float> upi(-PI, PI), u1(-1.0f, 1.0f);
  vector<float> x(M);
  vector<complex<float>> c(M), F(N);
  generate(x.begin(), x.end(), [&] { return upi(rng); }); // points in [-pi,pi)
  generate(c.begin(), c.end(), [&] { return complex<float>{u1(rng), u1(rng)}; });

  // copy x and c to the GPU, and make room for F; cuFloatComplex has the same layout
  // docs-start: simple1d1f-device
  thrust::device_vector<float> d_x          = x;
  thrust::device_vector<complex<float>> d_c = c, d_f = F;
  auto *dc = reinterpret_cast<cuFloatComplex *>(d_c.data().get());
  auto *df = reinterpret_cast<cuFloatComplex *>(d_f.data().get());
  // docs-end: simple1d1f-device
  // the cufinufftf prefix selects single precision
  // docs-start: simple1d1f-call
  int ier  = cufinufftf1d1(M, d_x.data().get(), dc, +1, tol, N, df, nullptr);
  // docs-end: simple1d1f-call
  // copy F back to the CPU
  // docs-start: simple1d1f-copy
  thrust::copy(d_f.begin(), d_f.end(), F.begin());
  // docs-end: simple1d1f-copy
  if (ier) return ier;

  constexpr int k = 1425; // check the answer just for this mode, in [-N/2,N/2)
  const complex<float> I(0.0f, 1.0f);
  complex<float> Ftest(0.0f, 0.0f);
  for (int j = 0; j < M; ++j) Ftest += c[j] * exp(I * float(k) * x[j]);
  float Fmax = 0.0f; // compute inf norm of F
  for (const auto &Fm : F) {
    const float a = abs(Fm);
    if (a > Fmax) Fmax = a;
  }
  const auto err = abs(F[k + N / 2] - Ftest) / Fmax; // k + N/2: index of freq mode k
  if (!(err <= 10 * tol)) {
    fprintf(stderr, "FAILED: rel err %.3g > %.3g\n", double(err), double(10 * tol));
    return 1;
  }
  printf("rel err %.3g\n", double(err));
  return 0;
}
