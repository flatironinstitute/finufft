// this is all you must include for the finufft lib...
#include <finufft.h>

// also used in this example...
#include <cassert>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

static const double PI = 3.141592653589793238462643383279502884;

int main(int argc, char *argv[])
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
  int M = 1e7; // number of nonuniform points
  int N = 1e7; // size of regular grid
  finufft_opts opts;
  finufft_default_opts(&opts);
  opts.spreadinterponly = 1;    // task: the following two control kernel used...
  double tol            = 1e-9; // tolerance for (real) kernel shape design only
  opts.upsampfac        = 2.0;  // pretend upsampling factor (really no upsampling)
  // opts.spread_kerevalmeth = 0;  // would be needed for any nonstd upsampfac

  std::complex<double> I = std::complex<double>(0.0, 1.0); // the imaginary unit
  std::vector<double> x(M);                                // input
  std::vector<std::complex<double>> c(M);                  // input
  std::vector<std::complex<double>> F(N);                  // output (spread to this array)

  // first spread M=1 single unit-strength at the origin, only to get its total mass...
  x[0]       = 0.0;
  c[0]       = 1.0;
  int unused = 1;
  int ier = finufft1d1(1, x.data(), c.data(), unused, tol, N, F.data(), &opts); // warm-up
  if (ier > 1) return ier;
  std::complex<double> kersum = 0.0;
  for (auto Fk : F) kersum += Fk; // kernel mass

  // Now generate random nonuniform points (x) and complex strengths (c)...
  for (int j = 0; j < M; ++j) {
    x[j] = PI * (2 * ((double)std::rand() / RAND_MAX) - 1); // uniform random in [-pi,pi)
    c[j] = 2 * ((double)std::rand() / RAND_MAX) - 1 +
           I * (2 * ((double)std::rand() / RAND_MAX) - 1);
  }

  opts.debug = 1;
  auto t0    = std::chrono::steady_clock::now(); // now spread with all M pts... (dir=1)
  ier      = finufft1d1(M, x.data(), c.data(), unused, tol, N, F.data(), &opts); // do it
  double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (ier > 1) return ier;
  std::complex<double> csum = 0.0; // tot input strength
  for (auto cj : c) csum += cj;
  std::complex<double> mass = 0.0; // tot output mass
  for (auto Fk : F) mass += Fk;
  double relerr = std::abs(mass - kersum * csum) / std::abs(mass);
  printf("1D spread-only, double-prec, %.3g s (%.3g NU pt/sec), ier=%d, mass err %.3g\n",
         t, M / t, ier, relerr);

  for (auto &Fk : F) Fk = std::complex<double>{1.0, 0.0}; // unit grid input
  opts.debug = 0;
  t0         = std::chrono::steady_clock::now(); // now interp to all M pts...  (dir=2)
  ier = finufft1d2(M, x.data(), c.data(), unused, tol, N, F.data(), &opts); // do it
  t   = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (ier > 1) return ier;
  csum = 0.0; // tot output
  for (auto cj : c) csum += cj;
  double maxerr = 0.0;
  for (auto cj : c) maxerr = std::max(maxerr, std::abs(cj - kersum));
  printf("1D interp-only, double-prec, %.3g s (%.3g NU pt/sec), ier=%d, max err %.3g\n",
         t, M / t, ier, maxerr / std::abs(kersum));
  return 0;
}
