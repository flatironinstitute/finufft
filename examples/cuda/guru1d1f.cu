/* Example of calling the cuFINUFFT guru interface from C++, using thrust device
   vectors, with a math test. GPU version of examples/guru1d1f.cpp.
   Single-precision version (see guru1d1.cu for double-precision). Barbone 11/12/26
   To compile, see README in this directory. Usage: ./guru1d1f
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
  constexpr int M     = 1e5;           // number of nonuniform points
  constexpr int N     = 1e4;           // number of modes
  constexpr float tol = 1e-3;          // desired accuracy

  constexpr int type = 1, dim = 1;     // 1d1
  constexpr int64_t Ns[3] = {N, 0, 0}; // guru describes mode array by vector [N1,N2..]
  constexpr int ntransf   = 1;         // we want to do a single transform at a time
  cufinufftf_plan plan;                // creates single-prec plan struct: note the "f"
  // nullptr means use default opts...
  int ier = cufinufftf_makeplan(type, dim, Ns, +1, ntransf, tol, &plan, nullptr);
  if (ier) return ier;

  mt19937 rng(12345);
  uniform_real_distribution<float> upi(-PI, PI), u1(-1.0f, 1.0f);

  // generate some random nonuniform points
  vector<float> x(M);
  generate(x.begin(), x.end(), [&] { return upi(rng); });
  thrust::device_vector<float> d_x = x; // copy x to the GPU
  ier = cufinufftf_setpts(plan, M, d_x.data().get(), nullptr, nullptr, 0, nullptr,
                          nullptr, nullptr);
  if (ier) return ier;

  // generate some complex strengths
  vector<complex<float>> c(M);
  generate(c.begin(), c.end(), [&] { return complex<float>{u1(rng), u1(rng)}; });

  // alloc output array for the Fourier modes, then do the transform
  vector<complex<float>> F(N);
  thrust::device_vector<complex<float>> d_c = c, d_f = F; // also c, and room for F
  auto *dc = reinterpret_cast<cuFloatComplex *>(d_c.data().get()); // same layout
  auto *df = reinterpret_cast<cuFloatComplex *>(d_f.data().get());
  ier      = cufinufftf_execute(plan, dc, df);                     // note "f"
  if (ier) return ier;

  // for fun, do another with same NU pts (no re-sorting), but new strengths...
  generate(c.begin(), c.end(), [&] { return complex<float>{u1(rng), u1(rng)}; });
  thrust::copy(c.begin(), c.end(), d_c.begin()); // copy the new strengths to the GPU
  ier = cufinufftf_execute(plan, dc, df);
  if (ier) return ier;
  thrust::copy(d_f.begin(), d_f.end(), F.begin()); // copy F back to the CPU

  cufinufftf_destroy(plan); // don't forget! done with transforms of this size

  // rest is math checking and reporting...
  constexpr int n = 1251; // check the answer just for this mode, must be in [-N/2,N/2)
  const complex<float> I(0.0f, 1.0f);
  complex<float> Ftest(0.0f, 0.0f);
  for (int j = 0; j < M; ++j) Ftest += c[j] * exp(I * float(n) * x[j]);
  float Fmax = 0.0f; // compute inf norm of F
  for (const auto &Fm : F) {
    const float a = abs(Fm);
    if (a > Fmax) Fmax = a;
  }
  const int nout = n + N / 2; // index in output array for freq mode n
  const auto err = abs(F[nout] - Ftest) / Fmax;
  if (!(err <= 10 * tol)) {
    fprintf(stderr, "FAILED: rel err %.3g > %.3g\n", double(err), double(10 * tol));
    return 1;
  }
  printf("rel err %.3g\n", double(err));
  return 0;
}
