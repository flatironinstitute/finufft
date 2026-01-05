/* test/tolsweep: pass-fail accuracy test for either float/double CPU FINUFFT
   that sweeps across full range of tolerances. Uses relative L2 error norms.
   Exists zero if success, nonzero upon failure.

   Based on Barbone's accuracy_test and Barnett's matlab/test/tolsweeptest.m
   The logic is taken from the latter. Barnett 1/4/26

   Todo
   1d for now
   get rid of inputs except:
   upsampfac, kerformula, showwarn, verbose (want worstfac...)?

*/

#include <cmath>
#include <cstdlib>
#include <finufft/test_defs.h>
#include <iomanip>
#include <iostream>
#include <vector>
// test utilities: direct DFT and norm helpers
#include "utils/dirft1d.hpp"
#include "utils/dirft2d.hpp"
#include "utils/dirft3d.hpp"
#include "utils/norms.hpp"
#include <finufft_common/kernel.h>

int main(int argc, char *argv[]) {

  // Define test problems, tolerance ranges, slack factors...
  BIGINT M  = 1000; // pick problem size: # sources
  BIGINT N  = 30;   // # modes
  int isign = +1;

  // one USF for now...
  double upsampfac = 2.0;
#ifdef SINGLE
  double floor = 1e-5;
#else
  double floor = 3e-14;
#endif
  double tolslack      = 5.0;                             // type 1 only for now
  double tolsperdecade = 8;
  double tolstep       = pow(10.0, -1.0 / tolsperdecade); // multiplicative tol step, <1

  // Defaults
  int kerformula = 0;
  int showwarn   = 0;
  int verbose    = 0;
  int debug      = 0;

  // If user asked for help, print usage and exit
  for (int ai = 1; ai < argc; ++ai) {
    if (std::string(argv[ai]) == "-h" || std::string(argv[ai]) == "--help") {
      std::cout << "Usage: " << argv[0] << " kerformula [showwarn [verbose [debug]]]]\n";
      std::cout
          << "  kerformula    : spread kernel formula (0:default, >0: for experts)\n";
      std::cout
          << "  showwarn      : whether to print warnings (0=silent default, 1=show)\n";
      std::cout << "  verbose       : 0 (default) silent, >0 print worstfac, etc\n";
      std::cout << "  debug         : passed to opts.debug\n";
      std::cout << "Example: " << argv[0] << " 0 1 1 0\n";
      return 0;
    }
  }
  if (argc > 1) sscanf(argv[1], "%d", &kerformula);
  if (argc > 2) sscanf(argv[2], "%d", &showwarn);
  if (argc > 3) sscanf(argv[3], "%d", &verbose);
  if (argc > 4) sscanf(argv[4], "%d", &debug);

  ***GOT HERE : loop over tols..

                // Build tolerance grid: exps from -1 down to -max_digits in 0.02 steps.
                // Use an integer-step loop to ensure the last exponent equals -max_digits
                // (avoids floating-point drift that could omit the final decade).
                std::vector<double>
                    exps;
  int nsteps = (int)std::round((max_digits - 1.0) / 0.02);
  for (int i = 0; i <= nsteps; ++i) exps.push_back(-1.0 - 0.02 * (double)i);
  const size_t NT = exps.size();
  std::vector<double> tols(NT);
  for (size_t t = 0; t < NT; ++t) tols[t] = pow(10.0, exps[t]);

  // Setup opts (will be passed to guru makeplan). We will use the plan
  // guru interface so we can override the plan's spread kernel selection
  // via the plan method `set_spread_kerformula` below.
  finufft_opts opts{};
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.upsampfac         = upsampfac;
  opts.spread_kerformula = kerformula;
  // set debug levels from command-line if requested
  opts.debug        = debug_level;
  opts.spread_debug = debug_level;
  // set whether to show warnings (user-controllable)
  opts.showwarn = showwarn;

  // For reproducibility use srand when not holding inputs
  srand(seed);

  std::vector<FLT> x(M);
  std::vector<CPX> c(M);
  std::vector<CPX> F(N);
  std::vector<CPX> fe(N);

  for (BIGINT j = 0; j < M; ++j) {
    x[j] = PI * randm11();
  }
  if (hold_inputs) {
    for (BIGINT j = 0; j < M; ++j) c[j] = crandm11();
  }

  int ier = 0;

  // Pass/fail tracking: we update these as each tolerance is tested.
  int npass = 0;
  int nfail = 0;
  std::vector<int> decade_fail(max_digits + 1, 0);
  std::vector<int> decade_total(max_digits + 1, 0);
  std::cout << std::scientific << std::setprecision(6);

  // Tunable slack multiplier: allow `slack * tol` as acceptable threshold.
  // `base_slack` can be tuned to be more/less permissive. When a tolerance
  // is close to machine precision (digit near `max_digits`) we increase the
  // slack to account for limits of floating-point resolution.
  const double base_slack = 3.0; // it fails in CI otherwsie
  for (size_t t = 0; t < NT; ++t) {
    double tol = tols[t];
    if (!hold_inputs) {
      for (BIGINT j = 0; j < M; ++j) {
        x[j] = PI * randm11();
        c[j] = crandm11();
      }
    }

    ier = FINUFFT1D1(M, x.data(), c.data(), isign, (FLT)tol, N, F.data(), &opts);
    if (ier > 1) {
      std::cerr << "accuracy_test: FINUFFT1D1 returned ier=" << ier << "\n";
      return ier;
    }

    // Compute exact result using direct DFT test utility and compute
    // relative L2 error using the test norms utilities.
    dirft1d1<BIGINT>(M, x, c, isign, N, fe);
    double rel_err = relerrtwonorm<BIGINT>(N, fe, F);

    // Compute integer digit from the precomputed exponents for exactness.
    int d = (int)std::round(-exps[t]);
    if (d < 1) d = 1;
    if (d > max_digits) d = max_digits;

    // If this tolerance is exactly a power-of-ten (1e-d), print the
    // achieved accuracy for that power-of-ten tolerance.
    if (std::fabs(-exps[t] - d) < 1e-12) {
      std::cout << "tol 1e-" << d << " achieved=" << rel_err << "\n";
    }

    // Compute the required threshold using a tunable slack multiplier.
    double slack = base_slack;
#ifdef SINGLE
    if (d == 6) slack *= 5.0;
    if (d == 7) slack *= 50.0;
#else
    if (d == 14) slack *= 1.10;
    if (d == 15) slack *= 13.0;
#endif

    const double req = tol * slack; // final acceptance threshold
    const bool pass  = (rel_err <= req);
    if (pass) {
      ++npass;
    } else {
      ++nfail;
      ++decade_fail[d];
      // Print failures immediately only in verbose mode (suppress PASSED lines to reduce
      // noise).
      if (verbose) {
        std::cout << "tol=" << tol << " req=" << req << " rel_err=" << rel_err
                  << " -> FAILED (interval=1e-" << d << ",1e-" << (d + 1) << ")\n";
      }
    }
    ++decade_total[d];

    // If the next sample is in a different interval (or we're at the last sample),
    // print the decade summary now so progress appears as we go.
    int next_d = -1;
    if (t + 1 < NT) next_d = static_cast<int>(std::round(-exps[t + 1]));
    if (next_d < 1) next_d = 1;
    if (next_d > max_digits) next_d = max_digits;
    if (next_d != d) {
      // Only print intermediate decade summaries in verbose mode; otherwise
      // the program will only emit the final SUMMARY to reduce noise.
      std::cout << "-- [1e-" << (d) << ", 1e-" << (d + 1)
                << "] summary: total=" << decade_total[d] << " failed=" << decade_fail[d]
                << "\n";
    }
  }

  std::cout << "\nSUMMARY: " << npass << " passed, " << nfail << " failed\n";
  return nfail != 0;
}
