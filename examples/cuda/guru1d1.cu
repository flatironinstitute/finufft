/* Example of calling the cuFINUFFT guru interface from C++, using thrust device
   vectors, with a math test. GPU version of examples/guru1d1.cpp.
   Double-precision version (see guru1d1f.cu for single-precision). Barbone 11/12/26
   To compile, see README in this directory. Usage: ./guru1d1
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
  constexpr int M      = 3e6;          // number of nonuniform points
  constexpr int N      = 1e6;          // number of modes
  constexpr double tol = 1e-9;         // desired accuracy

  constexpr int type = 1, dim = 1;     // 1d1
  constexpr int64_t Ns[3] = {N, 0, 0}; // guru describes mode array by vector [N1,N2..]
  constexpr int ntransf   = 1;         // we want to do a single transform at a time
  cufinufft_plan plan;                 // creates a plan struct
  // nullptr means use default opts...
  int ier = cufinufft_makeplan(type, dim, Ns, +1, ntransf, tol, &plan, nullptr);
  if (ier) return ier;

  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-PI, PI), u1(-1.0, 1.0);

  // generate some random nonuniform points
  vector<double> x(M);
  generate(x.begin(), x.end(), [&] { return upi(rng); });
  thrust::device_vector<double> d_x = x; // copy x to the GPU
  ier = cufinufft_setpts(plan, M, d_x.data().get(), nullptr, nullptr, 0, nullptr, nullptr,
                         nullptr);
  if (ier) return ier;

  // generate some complex strengths
  vector<complex<double>> c(M);
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });

  // alloc output array for the Fourier modes, then do the transform
  vector<complex<double>> F(N);
  thrust::device_vector<complex<double>> d_c = c, d_f = F; // also c, and room for F
  auto *dc = reinterpret_cast<cuDoubleComplex *>(d_c.data().get()); // same layout
  auto *df = reinterpret_cast<cuDoubleComplex *>(d_f.data().get());
  ier      = cufinufft_execute(plan, dc, df);
  if (ier) return ier;

  // for fun, do another with same NU pts (no re-sorting), but new strengths...
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });
  thrust::copy(c.begin(), c.end(), d_c.begin()); // copy the new strengths to the GPU
  ier = cufinufft_execute(plan, dc, df);
  if (ier) return ier;
  thrust::copy(d_f.begin(), d_f.end(), F.begin()); // copy F back to the CPU

  cufinufft_destroy(plan); // don't forget! done with transforms of this size

  // rest is math checking and reporting...
  constexpr int n = 142519; // check the answer just for this mode
  static_assert(n >= -N / 2.0 && n < N / 2.0); // ensure meaningful test
  const complex<double> I(0.0, 1.0);
  complex<double> Ftest(0.0, 0.0);
  for (int j = 0; j < M; ++j) Ftest += c[j] * exp(I * double(n) * x[j]);
  double Fmax = 0.0; // compute inf norm of F
  for (const auto &Fm : F) {
    const double a = abs(Fm);
    if (a > Fmax) Fmax = a;
  }
  const int nout = n + N / 2; // index in output array for freq mode n
  const auto err = abs(F[nout] - Ftest) / Fmax;
  if (!(err <= 10 * tol)) {
    fprintf(stderr, "FAILED: rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  printf("rel err %.3g\n", err);
  return 0;
}
