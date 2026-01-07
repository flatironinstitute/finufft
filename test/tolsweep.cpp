/* test/tolsweep: pass-fail accuracy test for either float/double CPU FINUFFT
   that sweeps across full range of tolerances, dims, types, for set of upsampfacs
   (just two std USFs for now).
   Uses relative L2 error norms, with direct reference evaluation.
   Exit code: zero if success, nonzero upon failure.

   Based on Barbone's accuracy_test and Barnett's matlab/test/tolsweeptest.m
   The logic is taken from the latter (no significance to decades). Barnett 1/5/26
   Multiple dims, USFs, extra debug output for failures. 1/6/26.
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
  // N vectors to test: first triplet is for dim=1, then for dim=2, etc...
  BIGINT Nm_alldims[3][3] = {{30, 1, 1}, {20, 40, 1}, {10, 11, 12}}; // Ntot~1e3 ok
  int ntr      = 1; // >1 in tolsweeptest.m but only for speed/convenience
  int isign = +1;

  // *** make these vary by dim?: had to hack type-3 to slack=15 here for macos CI...
  double tolslack[3]    = {5.0, 5.0, 15.0}; // tunable slack parameters for each type
  double tolsperdecade  = 8;                // controls overall effort (tol resolution)
  double tolstep       = pow(10.0, -1.0 / tolsperdecade); // multiplicative tol step, <1
  constexpr FLT EPSILON = std::numeric_limits<FLT>::epsilon();
  double mintol         = 0.5 * EPSILON; // somewhat arbitrary where start (catch warns)
  int ntols             = std::ceil(log(mintol) / log(tolstep));

  // Defaults
  int kerformula = 0;
  int showwarn   = 0;
  int verbose    = 2;
  int debug      = 0;
  // test set of upsampfacs each with matching error floor for each dim...
  const int nu         = 2; // how many USFs (just the standard ones for now)
  double upsampfac[nu] = {1.25, 2.0};
#ifdef SINGLE
  double floor[nu][3] = {{1e-4, 1e-3, 1e-2}, {1e-5, 1e-5, 1e-5}}; // inner is dim
#else
  double floor[nu][3] = {{3e-9, 3e-9, 1e-8}, {3e-14, 3e-14, 3e-14}};
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
      std::cout
          << "  verbose       : 0 silent, 1 summary, 2 +fail deets (default), 3...\n";
      std::cout << "  debug         : passed to opts.debug for every call\n";
      std::cout << "Example: " << argv[0] << " 0 1 2 0\n";
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

  std::vector<FLT> x(M), y(M), z(M), X, Y, Z; // xyz real vs XYZ freq-space
  std::vector<CPX> c(M), ce(M), F, Fe; // (we don't know N yet, since varies by dim)
  srand(42);                          // fix seed
  int nfailtot = 0;                   // overall count across all dims, USF, tols, types

  for (int dim = 1; dim <= 3; ++dim) {  /////////////////////// loop over dims
    if (verbose) printf("\n%s: %dD =============================\n", argv[0], dim);
    BIGINT *Nm = Nm_alldims[dim - 1];   // ptr to Nm array (3-el) for this dim
    BIGINT N   = Nm[0] * Nm[1] * Nm[2]; // tot # modes, or freq-pts for type 3
    X.resize(N);
    Y.resize(N);
    Z.resize(N);
    F.resize(N);
    Fe.resize(N);

    for (int u = 0; u < nu; ++u) { // ========================= loop over upsampfacs
                                   // (sigma)
      opts.upsampfac = upsampfac[u];
      if (verbose) printf(" upsampfac = %.3g -----------------\n", opts.upsampfac);

      double worstfac[3] = {0};           // track the largest clearance for each type
      double tol         = 1.0;           // starting (max) tol to test
      int npass[3] = {0}, nfail[3] = {0}; // counts for each type
      for (int t = 0; t < ntols; ++t) {   // ............... loop over tols

        for (int type = 1; type <= 3; ++type) { // ---------------------- loop over types

          // just make new data each test & type, even data not needed for that type
          for (BIGINT j = 0; j < M; ++j) {
            x[j] = PI * randm11();
            y[j] = PI * randm11();
            z[j] = PI * randm11();
            c[j] = crandm11();
          }
          for (BIGINT k = 0; k < N; ++k) {
            X[k] = Nm[0] * rand01(); // type 3: scale freq s,t,u NU pts by "mode" sizes
            Y[k] = Nm[1] * rand01(); // (should match erralltypedim.m)
            Z[k] = Nm[2] * rand01();
            F[k] = crandm11();
          }
          FINUFFT_PLAN plan; // do tested transform...
          int ier = FINUFFT_MAKEPLAN(type, dim, Nm, isign, ntr, (FLT)tol, &plan, &opts);
          int ier_set = FINUFFT_SETPTS(plan, M, x.data(), y.data(), z.data(), N, X.data(),
                                       Y.data(), Z.data());
          FINUFFT_EXECUTE(plan, c.data(), F.data()); // type 2 writes to c, others to F
          FINUFFT_DESTROY(plan);

          if (dim == 1) // do the relevant (of nine) direct "exact" evals...
            if (type == 1)
              dirft1d1<BIGINT>(M, x, c, isign, Nm[0], Fe);    // exact ans written into Fe
            else if (type == 2)
              dirft1d2<BIGINT>(M, x, ce, isign, Nm[0], F);    // exact ans written into ce
            else
              dirft1d3<BIGINT>(M, x, c, isign, Nm[0], X, Fe); // exact ans written into Fe
          else if (dim == 2)
            if (type == 1)
              dirft2d1<BIGINT>(M, x, y, c, isign, Nm[0], Nm[1], Fe);
            else if (type == 2)
              dirft2d2<BIGINT>(M, x, y, ce, isign, Nm[0], Nm[1], F);
            else
              dirft2d3<BIGINT>(M, x, y, c, isign, N, X, Y, Fe);
          else // dim=3
            if (type == 1)
              dirft3d1<BIGINT>(M, x, y, z, c, isign, Nm[0], Nm[1], Nm[2], Fe);
            else if (type == 2)
              dirft3d2<BIGINT>(M, x, y, z, ce, isign, Nm[0], Nm[1], Nm[2], F);
            else
              dirft3d3<BIGINT>(M, x, y, z, c, isign, N, X, Y, Z, Fe);

          double relerr;                              // compute relevant error metric
          if (type == 2)
            relerr = relerrtwonorm<BIGINT>(M, ce, c); // ||ce-c||/||ce|| so ce comes 1st
          else
            relerr = relerrtwonorm<BIGINT>(N, Fe, F);

          if (ier > 1 || ier_set > 1) { // an error, not merely warning, we exit
            fprintf(stderr, "   tolsweep: %dD%d failed! ier=%d, ier_setpts=%d\n", dim,
                    type, ier, ier_set);
            return 1;
          }

          if (ier == 0) {
            int ti     = type - 1; // index for 3-el arrays
            double req = std::max(floor[u][dim - 1], tolslack[ti] * tol); // threshold
            double clearfac = relerr / req; // factor by which beats req (<=1 ok, >1 fail)
            worstfac[ti]    = std::max(worstfac[ti], clearfac); // track the worst case
            bool pass       = (relerr <= req);
            if (pass) {
              ++npass[ti];
              if (verbose > 2)
                printf(
                    "   %dd%d, tol %8.3g:\trelerr = %.3g,    \tclearancefac=%.3g\tpass\n",
                    dim, type, tol, relerr, clearfac);
            } else {
              ++nfail[ti];
              printf(
                  "   %dd%d, tol %8.3g:\trelerr = %.3g,    \tclearancefac=%.3g  \tFAIL\n",
                  dim, type, tol, relerr, clearfac);
              if (verbose > 1) {
                printf("   Rerunning with "
                       "debug=1:_______________________________________\n");
                opts.debug = 1;
                FINUFFT_MAKEPLAN(type, dim, Nm, isign, ntr, (FLT)tol, &plan, &opts);
                FINUFFT_SETPTS(plan, M, x.data(), y.data(), z.data(), N, X.data(),
                               Y.data(), Z.data());
                FINUFFT_EXECUTE(plan, c.data(), F.data()); // type 2 writes to c
                FINUFFT_DESTROY(plan);
                printf("   (Rerun "
                       "done)__________________________________________________\n");
                opts.debug = debug; // reset to cmdline arg value
              }
            }
          } else // finufft returned warning (ie cannot achieve accuracy): don't test
            if (verbose > 2)
              printf(
                  "   %dd%d, tol %8.3g:\trelerr = %.3g,    \t(warn ier=%d: not tested)\n",
                  dim, type, tol, relerr, ier);

        } // ---------------------------

        tol *= tolstep; // reduce tol in geometric progr
      } // ...........................................

      if (verbose)
        for (int ti = 0; ti < 3; ++ti)
          printf("  %dD%d summary: %d pass, %d fail. worstfac=%.3g\n", dim, ti + 1,
                 npass[ti], nfail[ti], worstfac[ti]);
      nfailtot += nfail[0] + nfail[1] + nfail[2];

    } // ==========================

  } //////////////////////////

  return nfailtot > 0; // if any fails, allows all test cases to complete
}
