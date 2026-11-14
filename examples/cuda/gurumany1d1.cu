/* Example of calling the cuFINUFFT guru interface from C++, using thrust device
   vectors, with a math test: a stack of 1D type 1 transforms in one execute
   call. GPU version of examples/gurumany1d1.cpp. Barbone 11/12/26
   To compile, see README in this directory. Usage: ./gurumany1d1
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
  constexpr int M      = 2e5;  // number of nonuniform points
  constexpr int N      = 1e5;  // number of modes
  constexpr double tol = 1e-9; // desired accuracy
  constexpr int ntrans = 100;  // request a bunch of transforms in the execute
  constexpr int isign  = +1;   // sign of i in the transform math definition

  constexpr int type = 1, dim = 1;     // 1d1
  constexpr int64_t Ns[3] = {N, 0, 0}; // guru describes mode array by vector [N1,N2..]
  cufinufft_plan plan;                 // creates a plan struct (nullptr: default opts)
  int ier = cufinufft_makeplan(type, dim, Ns, isign, ntrans, tol, &plan, nullptr);
  if (ier) return ier;

  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-PI, PI), u1(-1.0, 1.0);

  // generate random nonuniform points and pass to cuFINUFFT
  vector<double> x(M);
  generate(x.begin(), x.end(), [&] { return upi(rng); });
  thrust::device_vector<double> d_x = x; // copy x to the GPU
  ier = cufinufft_setpts(plan, M, d_x.data().get(), nullptr, nullptr, 0, nullptr, nullptr,
                         nullptr);
  if (ier) return ier;

  // generate ntrans complex strength vectors each of length M (the slow bit!)
  vector<complex<double>> c(M * ntrans); // plain contiguous storage
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });

  // alloc output array for the Fourier modes, then do the transform
  vector<complex<double>> F(N * ntrans);
  thrust::device_vector<complex<double>> d_c = c, d_f = F; // also c, and room for F
  auto *dc = reinterpret_cast<cuDoubleComplex *>(d_c.data().get()); // same layout
  auto *df = reinterpret_cast<cuDoubleComplex *>(d_f.data().get());
  ier      = cufinufft_execute(plan, dc, df);
  if (ier) return ier;
  thrust::copy(d_f.begin(), d_f.end(), F.begin()); // copy F back to the CPU

  // could now change c, do another execute, do another setpts, execute, etc...
  cufinufft_destroy(plan); // don't forget! we're done with transforms of this size

  // rest is math checking and reporting...
  constexpr int k     = 42519;                 // check the answer just for this mode
  constexpr int trans = ntrans - 1;            // ...testing in just this transform
  static_assert(k >= -N / 2.0 && k < N / 2.0); // ensure meaningful test
  static_assert(trans >= 0 && trans < ntrans);
  const complex<double> I(0.0, 1.0);
  complex<double> Ftest(0.0, 0.0);
  for (int j = 0; j < M; ++j)
    Ftest += c[j + M * trans] * exp(I * double(k) * x[j]); // c offset to trans
  double Fmax = 0.0; // compute inf norm of F for selected transform
  for (const auto &Fm : F) {
    const double a = abs(Fm);
    if (a > Fmax) Fmax = a;
  }
  const int nout = k + N / 2 + N * trans; // freq mode k index in the trans
  const auto err = abs(F[nout] - Ftest) / Fmax;
  if (!(err <= 10 * tol)) {
    fprintf(stderr, "FAILED: rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  printf("rel err %.3g\n", err);
  return 0;
}
