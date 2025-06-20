/* Tester for 1D spread/interp only tasks. The math tests are:
   1) for spread, check sum of spread kernel masses is as expected from one
   times the strengths (ie testing the zero-frequency component in NUFFT).
   2) for interp, check each interp kernel mass is the same as from one.
   Without knowing the kernel, this is about all that can be done, basically test
   that arbitrary kernel translate samples sum to the same value within O(tol).
   (Better math tests would be, ironically, to wrap the spreader/interpolator
   into a NUFFT and test that :) But we already have that in FINUFFT.)

   Barnett 1/8/25, based on ../examples/spreadinterponly1d and finufft1d_test
*/
#include <finufft/test_defs.h>
using namespace std;
using namespace finufft::utils;

const char *help[] = {"Tester for FINUFFT in 1d, spread/interp only, either precision.",
                      "",
                      "Usage: spreadinterp1d_test Ngrid Nsrc [tol [debug [spread_sort "
                      "[upsampfac [errfail]]]]]",
                      "\teg:\tspreadinterp1d_test 1e6 1e6 1e-6 1 2 2.0 1e-5",
                      "\tnotes:\tif errfail present, exit code 1 if any error > errfail",
                      NULL};

int main(int argc, char *argv[]) {
  BIGINT M, N;          // M = # nonuniform pts, N = # regular grid pts
  double w, tol = 1e-6; // default: used for kernel shape design only
  double errfail = INFINITY;
  FLT errmax     = 0;
  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts); // put defaults in opts
  opts.spreadinterponly = 1;   // this task
  if (argc < 3 || argc > 8) {
    for (int i = 0; help[i]; ++i) fprintf(stderr, "%s\n", help[i]);
    return 2;
  }
  sscanf(argv[1], "%lf", &w);
  N = (BIGINT)w;
  sscanf(argv[2], "%lf", &w);
  M = (BIGINT)w;
  if (argc > 3) sscanf(argv[3], "%lf", &tol);
  if (argc > 4) sscanf(argv[4], "%d", &opts.debug);
  opts.spread_debug = (opts.debug > 1) ? 1 : 0; // see output from spreader
  if (argc > 5) sscanf(argv[5], "%d", &opts.spread_sort);
  if (argc > 6) {
    sscanf(argv[6], "%lf", &w);
    opts.upsampfac = (FLT)w; // pretend upsampling for kernel (really no upsampling)
  }
  if (argc > 7) sscanf(argv[7], "%lf", &errfail);

  vector<FLT> x(M); // NU pts
  vector<CPX> c(M); // their strengths
  vector<CPX> F(M); // values on regular I/O grid (not Fourier coeffs for this task!)

  // first spread M=1 single unit-strength at the origin, to get its total mass...
  x[0]       = 0.0;
  c[0]       = 1.0;
  int unused = +1;
  int ier    = FINUFFT1D1(1, x.data(), c.data(), unused, tol, N, F.data(), &opts);
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  }
  CPX kersum = 0.0;
  for (auto Fk : F) kersum += Fk; // one kernel mass

  // generate random nonuniform points (x) and complex strengths (c)
#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM();   // needed for parallel random #s
#pragma omp for schedule(static, TEST_RANDCHUNK) // static => non-stochastic
    for (BIGINT j = 0; j < M; ++j) {
      x[j] = PI * randm11r(&se);                 // fills [-pi,pi)
      c[j] = crandm11r(&se);
    }
  }
  printf("spread-only test 1d:\n"); // ............................................
  CNTime timer;
  timer.start();                    // c input, F output...
  ier      = FINUFFT1D1(M, x.data(), c.data(), unused, tol, N, F.data(), &opts);
  double t = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  }
  printf("\t%lld NU pts spread to %lld grid in %.3g s \t%.3g NU pts/s\n", (long long)M,
         (long long)N, t, M / t);
  CPX csum = 0.0; // tot input strength
  for (auto cj : c) csum += cj;
  CPX mass = 0.0; // tot output mass
  for (auto Fk : F) mass += Fk;
  FLT relerr = abs(mass - kersum * csum) / abs(mass);
  printf("\trel mass err %.3g\n", relerr);
  errmax = max(relerr, errmax);

  printf("interp-only test 1d:\n"); // ............................................
  for (auto &Fk : F) Fk = complex<double>{1.0, 0.0}; // unit grid input
  timer.restart();                                   // F input, c output...
  ier = FINUFFT1D2(M, x.data(), c.data(), unused, tol, N, F.data(), &opts);
  t   = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  }
  printf("\t%lld NU pts interp from %lld grid in %.3g s \t%.3g NU pts/s\n", (long long)M,
         (long long)N, t, M / t);
  csum = 0.0; // tot output
  for (auto cj : c) csum += cj;
  FLT superr = 0.0;
  for (auto cj : c) superr = max(superr, abs(cj - kersum));
  FLT relsuperr = superr / abs(kersum);
  printf("\trel sup err %.3g\n", relsuperr);
  errmax = max(relsuperr, errmax);

  return (errmax > (FLT)errfail);
}
