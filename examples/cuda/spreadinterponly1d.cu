/* Example of calling the cuFINUFFT library from C++, using thrust device
   vectors, with math tests. GPU version of examples/spreadinterponly1d.cpp.
   Spread/interp-only tasks via opts.gpu_spreadinterponly=1. Complex I/O
   arrays, but recall the kernel is real. Barbone 11/12/26.
   To compile, see README in this directory. Usage: ./spreadinterponly1d
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
  constexpr int64_t M = 1e7; // number of nonuniform points
  constexpr int64_t N = 1e7; // size of regular grid

  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.gpu_spreadinterponly = 1;
  opts.upsampfac            = 2.0;    // pretend upsampling factor (really none)
  constexpr double tol      = 1e-9;   // tolerance for (real) kernel shape design only

  vector<double> x(M);                // input
  vector<complex<double>> c(M), F(N); // c: input; F: output

  // first spread M=1 single unit-strength at the origin, only to get its total mass...
  x[0]                                       = 0.0;
  c[0]                                       = 1.0;
  thrust::device_vector<double> d_x          = x;          // copy x to the GPU
  thrust::device_vector<complex<double>> d_c = c, d_f = F; // also c, and room for F
  auto *dc   = reinterpret_cast<cuDoubleComplex *>(d_c.data().get()); // same layout
  auto *df   = reinterpret_cast<cuDoubleComplex *>(d_f.data().get());
  int unused = 1;
  int ier = cufinufft1d1(1, d_x.data().get(), dc, unused, tol, N, df, &opts); // warm-up
  if (ier) return ier;
  thrust::copy(d_f.begin(), d_f.end(), F.begin()); // copy F back to the CPU
  complex<double> kersum = 0.0;
  for (const auto &v : F) kersum += v;             // kernel mass

  // Now generate random nonuniform points (x) and complex strengths (c)...
  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-PI, PI), u1(-1.0, 1.0);
  generate(x.begin(), x.end(), [&] { return upi(rng); });
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });
  d_x = x;
  d_c = c;

  ier = cufinufft1d1(M, d_x.data().get(), dc, unused, tol, N, df, &opts); // spread
  if (ier) return ier;
  thrust::copy(d_f.begin(), d_f.end(), F.begin()); // copy F back to the CPU
  complex<double> csum = 0.0;                      // tot input strength
  for (auto cj : c) csum += cj;
  complex<double> mass = 0.0;                      // tot output mass
  for (auto Fk : F) mass += Fk;
  double relerr = abs(mass - kersum * csum) / abs(mass);
  printf("1D spread-only, double-prec, ier=%d, mass err %.3g\n", ier, relerr);

  fill(F.begin(), F.end(), complex<double>{1.0, 0.0}); // unit grid input
  d_f = F;
  ier = cufinufft1d2(M, d_x.data().get(), dc, unused, tol, N, df, &opts); // interp
  if (ier) return ier;
  thrust::copy(d_c.begin(), d_c.end(), c.begin()); // copy c back to the CPU
  csum = 0.0;                                      // tot output
  for (auto cj : c) csum += cj;
  double maxerr = 0.0;
  for (auto cj : c) maxerr = max(maxerr, abs(cj - kersum));
  printf("1D interp-only, double-prec, ier=%d, max err %.3g\n", ier,
         maxerr / abs(kersum));
  // max() skips a NaN element, but csum carries it
  if (!(relerr <= 10 * tol && maxerr <= 10 * tol * abs(kersum) && !isnan(abs(csum))))
    return 1;
  return 0;
}
