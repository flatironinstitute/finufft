// this is all you must include for the finufft lib...
#include <finufft.h>

// also used in this example...
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>
using namespace std;

constexpr double pi = 3.14159265358979323846;
using namespace std::chrono;

int main()
/* Example of double-prec spread/interp only tasks, with basic math tests.
   Complex I/O arrays, but recall the kernel is real.  Barnett 1/8/25.

   The math tests are:
   1) for spread, check sum of spread kernel masses is as expected from sum
   of strengths (ie testing the zero-frequency component in NUFFT).
   2) for interp, check each interp kernel mass is the same as from one.

   Without knowing the kernel, this is about all that can be done!
   (Better math tests would be, ironically, to wrap the spreader/interpolator
   into a NUFFT and test that :) But we already have that in FINUFFT.)

   To compile, see README. Usage: ./spreadinterponly1d
   See: spreadtestnd for usage of internal (non FINUFFT-API) spread/interp.
*/
{
  constexpr int M = 1e7; // number of nonuniform points
  constexpr int N = 1e7; // size of regular grid
  finufft_opts opts;
  finufft_default_opts(&opts);
  opts.spreadinterponly = 1;    // task: the following two control kernel used...
  constexpr double tol  = 1e-9; // tolerance for (real) kernel shape design only
  opts.upsampfac        = 2.0;  // pretend upsampling factor (really no upsampling)

  vector<double> x(M);          // input
  vector<complex<double>> c(M); // input
  vector<complex<double>> F(N); // output (spread to this array)

  // first spread M=1 single unit-strength at the origin, only to get its total mass...
  x[0]       = 0.0;
  c[0]       = 1.0;
  int unused = 1;
  int ier = finufft1d1(1, x.data(), c.data(), unused, tol, N, F.data(), &opts); // warm-up
  if (ier > 0) return ier;
  const auto kersum = reduce(F.begin(), F.end()); // kernel mass

  // Now generate random nonuniform points (x) and complex strengths (c)...
  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);
  generate(x.begin(), x.end(), [&] { return upi(rng); });
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });

  opts.debug = 1;
  auto t0    = steady_clock::now(); // now spread with all M pts... (dir=1)
  ier      = finufft1d1(M, x.data(), c.data(), unused, tol, N, F.data(), &opts); // do it
  double t = (steady_clock::now() - t0) / 1.0s;
  if (ier > 0) return ier;
  const auto csum   = reduce(c.begin(), c.end()); // tot input strength
  const auto mass   = reduce(F.begin(), F.end()); // tot output mass
  const auto relerr = abs(mass - kersum * csum) / abs(mass);
  printf("1D spread-only, double-prec, %.3g s (%.3g NU pt/sec), ier=%d, mass err %.3g\n",
         t, M / t, ier, relerr);

  fill(F.begin(), F.end(), complex<double>{1.0, 0.0}); // unit grid input
  opts.debug = 0;
  t0         = steady_clock::now(); // now interp to all M pts...  (dir=2)
  ier = finufft1d2(M, x.data(), c.data(), unused, tol, N, F.data(), &opts); // do it
  t   = (steady_clock::now() - t0) / 1.0s;
  if (ier > 0) return ier;
  vector<double> terms(M);
  transform(c.begin(), c.end(), terms.begin(), [&](auto cj) { return abs(cj - kersum); });
  const auto maxerr = *max_element(terms.begin(), terms.end());
  printf("1D interp-only, double-prec, %.3g s (%.3g NU pt/sec), ier=%d, max err %.3g\n",
         t, M / t, ier, maxerr / abs(kersum));
  return !(relerr < 10 * tol && maxerr < 10 * tol * abs(kersum));
}
