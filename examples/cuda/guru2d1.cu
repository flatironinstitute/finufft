/* 2D type 1 guru interface example of calling the cuFINUFFT library from C++,
   using thrust device vectors, with a math test. Double precision. GPU version
   of examples/guru2d1.cpp.
   To compile, see README in this directory. Usage: ./guru2d1
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
  constexpr int M      = 2e6;  // number of nonuniform points
  constexpr double tol = 1e-6; // desired accuracy
  cufinufft_opts opts;         // demo how to change options away from defaults
  cufinufft_default_opts(&opts);
  opts.upsampfac = 1.25;       // match the CPU example

  // generate random non-uniform points (x,y) in [-PI,PI] and strengths (c)
  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-PI, PI), u1(-1.0, 1.0);
  vector<double> x(M), y(M);
  vector<complex<double>> c(M);
  generate(x.begin(), x.end(), [&] { return upi(rng); });
  generate(y.begin(), y.end(), [&] { return upi(rng); });
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });

  const int N1 = 1000, N2 = 2000; // numbers of output Fourier modes
  vector<complex<double>> F(N1 * N2);

  // step 1: make a plan...
  constexpr int type = 1, dim = 2, ntrans = 1; // you could also do ntrans>1
  const int64_t Ns[] = {N1, N2};               // N1,N2 as 64-bit int array
  cufinufft_plan plan;
  int ier = cufinufft_makeplan(type, dim, Ns, +1, ntrans, tol, &plan, &opts);
  if (ier) return ier;

  // step 2: send in M nonuniform points (just x, y in this case)...
  thrust::device_vector<double> d_x = x, d_y = y;          // copy x, y to the GPU
  thrust::device_vector<complex<double>> d_c = c, d_f = F; // also c, and room for F
  auto *dc = reinterpret_cast<cuDoubleComplex *>(d_c.data().get()); // same layout
  auto *df = reinterpret_cast<cuDoubleComplex *>(d_f.data().get());
  ier = cufinufft_setpts(plan, M, d_x.data().get(), d_y.data().get(), nullptr, 0, nullptr,
                         nullptr, nullptr);
  if (ier) return ier;
  // step 3: do the planned transform to the c strength data, output to F...
  ier = cufinufft_execute(plan, dc, df);
  if (ier) return ier;
  // ... you could now send in new points, and/or do transforms with new c data
  // ...
  thrust::copy(d_f.begin(), d_f.end(), F.begin()); // copy F back to the CPU
  // step 4: free the memory used by the plan...
  cufinufft_destroy(plan);

  // rest is math checking and reporting...
  const complex<double> I(0.0, 1.0);
  const int k1 = round(0.45 * N1);  // check the answer for mode frequency (k1,k2)
  const int k2 = round(-0.35 * N2); // each in [-N/2,N/2)
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
