/* test/tolsweep: pass-fail accuracy test for either float/double CPU FINUFFT
   that sweeps across full range of tolerances. Uses relative L2 error norms.
   Exists zero if success, nonzero upon failure.

   Based on Barbone's accuracy_test and Barnett's matlab/test/tolsweeptest.m
   The logic is taken from the latter (no significance to decades). Barnett 1/5/26
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

int main(int argc, char *argv[]) {

  // Define test problems, tolerance ranges, slack factors...
  BIGINT M  = 1000; // pick problem size: # sources
  BIGINT N  = 30;   // # modes
  int isign = +1;

  double tolslack       = 5.0; // tunable slack parameter(s); type 1 only for now
  double tolsperdecade = 8;
  double tolstep       = pow(10.0, -1.0 / tolsperdecade); // multiplicative tol step, <1
  constexpr FLT EPSILON = std::numeric_limits<FLT>::epsilon();
  double mintol         = 0.5 * EPSILON; // somewhat arbitrary where start
  int ntols             = std::ceil(log(mintol) / log(tolstep));

  // Defaults
  int kerformula = 0;
  int showwarn   = 0;
  int verbose    = 0;
  int debug      = 0;
  // one USF for now, matching error floor...
  double upsampfac = 0.0;
#ifdef SINGLE
  double floor = 1e-5;
#else
  double floor = 3e-14;
#endif

  // If user asked for help, print usage and exit
  for (int ai = 1; ai < argc; ++ai) {
    if (std::string(argv[ai]) == "-h" || std::string(argv[ai]) == "--help") {
      std::cout << "Usage: " << argv[0]
                << " kerformula [showwarn [verbose [debug [sigma [floor]]]]]]\n";
      std::cout
          << "  kerformula    : spread kernel formula (0:default, >0: for experts)\n";
      std::cout
          << "  showwarn      : whether to print warnings (0=silent default, 1=show)\n";
      std::cout << "  verbose       : 0 (default) silent, >0 print worstfac, etc\n";
      std::cout << "  debug         : passed to opts.debug\n";
      std::cout << "  sigma         : upsampling factor (default 0; passed to "
                   "opts.upsampfac)\n";
      std::cout << "  floor         : minimum rel err (default around 1e2 * eps_mach)\n";
      std::cout << "Example: " << argv[0] << " 0 1 1 0 2.0 3e-14\n";
      return 0;
    }
  }
  if (argc > 1) sscanf(argv[1], "%d", &kerformula);
  if (argc > 2) sscanf(argv[2], "%d", &showwarn);
  if (argc > 3) sscanf(argv[3], "%d", &verbose);
  if (argc > 4) sscanf(argv[4], "%d", &debug);
  if (argc > 5) sscanf(argv[5], "%lf", &upsampfac);
  if (argc > 6) sscanf(argv[6], "%lf", &floor);

  finufft_opts opts{};
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.upsampfac         = upsampfac;
  opts.spread_kerformula = kerformula;
  opts.debug             = debug;
  opts.showwarn = showwarn;

  std::vector<FLT> x(M);
  std::vector<CPX> c(M);
  std::vector<CPX> F(N);
  std::vector<CPX> fe(N);

  srand(42);                        // seed
  double worstfac = 0.0, tol = 1.0; // init tol
  int npass = 0, nfail = 0;
  for (int t = 0; t < ntols; ++t) { // ............... loop over tols

    // make new rand data each test
    for (BIGINT j = 0; j < M; ++j) {
      x[j] = PI * randm11();
      c[j] = crandm11();
    }

    // write into F
    int ier = FINUFFT1D1(M, x.data(), c.data(), isign, (FLT)tol, N, F.data(), &opts);
    if (ier > 1) {
      std::cerr << "tolsweep: FINUFFT1D1 failed with ier=" << ier << "\n";
      return ier;
    }
    dirft1d1<BIGINT>(M, x, c, isign, N, fe); // exact ans written into fe
    double relerr = relerrtwonorm<BIGINT>(N, fe, F);

    if (ier == 0) {
      double req      = std::max(floor, tolslack * tol); // acceptance threshold
      double clearfac = relerr / req; // factor by which beats req (<=1 ok, >1 fail)
      worstfac        = std::max(worstfac, clearfac); // track the worst case
      bool pass       = (relerr <= req);
      if (pass) {
        ++npass;
        if (verbose > 1)
          printf("\ttol %8.3g:\trelerr = %.3g,    \tclearancefac=%.3g\tpass\n", tol,
                 relerr, clearfac);
      } else {
        ++nfail;
        printf("\ttol %8.3g:\trelerr = %.3g,    \tclearancefac=%.3g\tFAIL\n", tol, relerr,
               clearfac);
      }
    } else // finufft returned warning, assumed cannot achieve accuracy
      if (verbose > 1)
        printf("\ttol %8.3g:\trelerr = %.3g,    \t(warn ier=%d: not tested)\n", tol,
               relerr, ier);

    tol *= tolstep;
  } // ...........................................

  if (verbose)
    printf("tolsweep 1d1: %d pass, %d fail. worstfac=%.3g\n", npass, nfail, worstfac);

  return nfail != 0;
}
