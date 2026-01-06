/* test/tolsweep: pass-fail accuracy test for either float/double CPU FINUFFT
   that sweeps across full range of tolerances. Uses relative L2 error norms.
   Exists zero if success, nonzero upon failure.

   Based on Barbone's accuracy_test and Barnett's matlab/test/tolsweeptest.m
   The logic is taken from the latter (no significance to decades). Barnett 1/5/26
   1D only for now, since low-upsampfac rdyn^dim effects need fitting.
*/

#include <cmath>
#include <cstdlib>
#include <finufft/test_defs.h>
#include <iomanip>
#include <iostream>
#include <vector>
// test utilities: direct DFT and norm helpers
#include "utils/dirft1d.hpp"
// #include "utils/dirft2d.hpp"
// #include "utils/dirft3d.hpp"
#include "utils/norms.hpp"

int main(int argc, char *argv[]) {

  // Define test problems, tolerance ranges, slack factors...
  int dim   = 1;    // do not change yet (no alloc for dim>1)
  BIGINT M  = 1000; // pick problem size: # sources
  BIGINT N  = 30;   // # modes
  int isign = +1;

  double tolslack[3]    = {5.0, 5.0, 10.0}; // tunable slack parameters for each type
  double tolsperdecade = 8;
  double tolstep       = pow(10.0, -1.0 / tolsperdecade); // multiplicative tol step, <1
  constexpr FLT EPSILON = std::numeric_limits<FLT>::epsilon();
  double mintol         = 0.5 * EPSILON; // somewhat arbitrary where start
  int ntols             = std::ceil(log(mintol) / log(tolstep));

  // Defaults
  int kerformula = 0;
  int showwarn   = 0;
  int verbose    = 1;
  int debug      = 0;
  // test set of upsampfacs each with matching error floor...
  const int nu = 2;    // how many USFs
  double upsampfac[nu] = {1.25, 2.0};
#ifdef SINGLE
  double floor[nu] = {1e-4, 1e-5};
#else
  double floor[nu] = {3e-9, 3e-14};
#endif

  // If user asked for help, print usage and exit
  for (int ai = 1; ai < argc; ++ai) {
    if (std::string(argv[ai]) == "-h" || std::string(argv[ai]) == "--help") {
      std::cout << "Usage: " << argv[0]
                << " [kerformula [showwarn [verbose [debug]]]]\n";
      std::cout
          << "  kerformula    : spread kernel formula (0:default, >0: for experts)\n";
      std::cout
          << "  showwarn      : whether to print warnings (0=silent default, 1=show)\n";
      std::cout << "  verbose       : 0 silent, 1 summary (default), 2 every test...\n";
      std::cout << "  debug         : passed to opts.debug\n";
      std::cout << "Example: " << argv[0] << " 0 1 1 0\n";
      return 0;
    }
  }
  if (argc > 1) sscanf(argv[1], "%d", &kerformula);
  if (argc > 2) sscanf(argv[2], "%d", &showwarn);
  if (argc > 3) sscanf(argv[3], "%d", &verbose);
  if (argc > 4) sscanf(argv[4], "%d", &debug);

  finufft_opts opts{};
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.spread_kerformula = kerformula;
  opts.debug             = debug;
  opts.showwarn = showwarn;

  std::vector<FLT> x(M), s(N);
  std::vector<CPX> c(M), ce(M), F(N), Fe(N);

  srand(42);                          // fix seed

  for (int u=0; u<nu; ++u) { // ========================== loop over upsampfacs (sigma)
    opts.upsampfac         = upsampfac[u];
    if (verbose) printf("tolsweep: upsampfac = %.3g...\n",opts.upsampfac);

    double worstfac[3] = {0};           // track the largest clearance for each type
    double tol         = 1.0;           // starting (max) tol to test
    int npass[3] = {0}, nfail[3] = {0}; // counts for each type
    for (int t = 0; t < ntols; ++t) { // ............... loop over tols

      for (int type = 1; type <= 3; ++type) { // ---------------------- loop over types

        // just make new rand data each test and type, even data not needed for that type
        for (BIGINT j = 0; j < M; ++j) {
          x[j] = PI * randm11();
          c[j] = crandm11();
        }
        for (BIGINT k = 0; k < N; ++k) {
          s[k] = PI * randm11();
          F[k] = crandm11();
        }
        // do tested transform and direct version...
        int ier;                                   // things needed in this scope
        double relerr;
        if (type == 1) {                           // writes into F
          ier = FINUFFT1D1(M, x.data(), c.data(), isign, (FLT)tol, N, F.data(), &opts);
          dirft1d1<BIGINT>(M, x, c, isign, N, Fe); // exact ans written into Fe
        } else if (type == 2) {                    // write into c
          ier = FINUFFT1D2(M, x.data(), c.data(), isign, (FLT)tol, N, F.data(), &opts);
          dirft1d2<BIGINT>(M, x, ce, isign, N, F); // exact ans written into ce
        } else {                                   // type 3, write into F
          ier = FINUFFT1D3(M, x.data(), c.data(), isign, (FLT)tol, N, s.data(), F.data(),
                           &opts);
          dirft1d3<BIGINT>(M, x, c, isign, N, s, Fe); // exact ans written into Fe
        }
        if (type == 2)                                // compute relevant error metric
          relerr = relerrtwonorm<BIGINT>(M, ce, c);
        else
          relerr = relerrtwonorm<BIGINT>(N, Fe, F);

        if (ier > 1) { // error not merely warning
          fprintf(stderr, "  tolsweep: %dD%d failed! ier=%d\n", dim, type, ier);
          return ier;
        }

        if (ier == 0) {
          int ti          = type - 1;                            // index for 3-el arrays
          double req      = std::max(floor[u], tolslack[ti] * tol); // acceptance threshold
          double clearfac = relerr / req; // factor by which beats req (<=1 ok, >1 fail)
          worstfac[ti]    = std::max(worstfac[ti], clearfac); // track the worst case
          bool pass       = (relerr <= req);
          if (pass) {
            ++npass[ti];
            if (verbose > 1)
              printf("  %dD%d, tol %8.3g:\trelerr = %.3g,    \tclearancefac=%.3g\tpass\n",
                     dim, type, tol, relerr, clearfac);
          } else {
            ++nfail[ti];
            printf("  %dD%d, tol %8.3g:\trelerr = %.3g,    \tclearancefac=%.3g\tFAIL\n",
                   dim, type, tol, relerr, clearfac);
            printf("  Rerunning with debug=1...........................................\n");
            // *** TO DO

            printf("  (Rerun done) ....................................................\n");
          }
        } else // finufft returned warning (namely cannot achieve accuracy): don't test acc
          if (verbose > 1)
            printf("  %dD%d, tol %8.3g:\trelerr = %.3g,    \t(warn ier=%d: not tested)\n",
                   dim, type, tol, relerr, ier);

      } // ---------------------------

      tol *= tolstep;
    } // ...........................................

    if (verbose)
      for (int ti = 0; ti < 3; ++ti)
        printf(" 1d%d: %d pass, %d fail. worstfac=%.3g\n", ti + 1, npass[ti],
               nfail[ti], worstfac[ti]);

    int nfailtot = nfail[0] + nfail[1] + nfail[2];
    if (nfailtot>0) return 1;

  }  // ==========================
  return 0;
}
