/* Test executable for the 1D, 2D, or 3D spreader, both directions.
 * This version uses the public FINUFFT guru API to exercise the spreader
 * behavior and checks correctness via kernel sums.

 * Barbone, using the public finufft API 11/07/2025.
 * Barnett 1/12/26: simplified to remove misleading/deprecated opts,
   fix erroneous NU pts change w/o setpts! Added both dir=1,2 as docs claimed.
 * Note that there is a rounding stochastic error of sqrt(M)*e_mach, which is
   apparent in single-prec (eg for M=6 you can bottom out at 3e-5 rel err).

 Todo: Add some more args + update screen output like spreadtestnd.cpp.
 */
#include "finufft/finufft_utils.hpp"
#include <finufft/test_defs.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

using namespace finufft::utils; // for timer

/* clang-format off */
void usage(void) {
  printf(
      "Speed tester for spread/interp sweeping over full tolerance range.\n\n"
      "usage:\t./spreadtestndall{f} dims [M N [sort [upsampfac]]]\n"
      "\twhere the suffix f is for single (else double prec), dims=1,2 or 3\n"
      "\tM=# nonuniform pts\n"
      "\tN=# uniform pts (rough total; per-dim N = round(N^(1/d)))\n"
      "\tsort=0 (never sort NU pts), 1 (always sort), or 2 (auto; default)\n"
      "\tupsampfac>1.0: sigma upsampling factor (typ range 1.2 to 2.5)\n"
      "\n"
      "example: ./spreadtestndall 3 1e6 1e6 2 1.25\n");
}
/* clang-format on */

int main(int argc, char *argv[]) {
  int d;
  double w;
  // Cmd line args & their defaults...
  BIGINT M        = 1e6; // default # NU pts
  BIGINT roughNg  = 1e6; // default # U pts
  int sort        = 2;   // spread_sort
  FLT upsampfac   = 2.0; // default

  if (argc < 2 || argc == 3 || argc > 6) {
    usage();
    return (argc > 1);
  }
  sscanf(argv[1], "%d", &d);
  if (d < 1 || d > 3) {
    printf("d must be 1, 2 or 3!\n");
    usage();
    return 1;
  }
  if (argc > 2) {
    sscanf(argv[2], "%lf", &w);
    M = (BIGINT)w; // to read "1e6" right
    if (M < 1) {
      printf("M (# NU pts) must be positive!\n");
      usage();
      return 1;
    }
    sscanf(argv[3], "%lf", &w);
    roughNg = (BIGINT)w;
    if (roughNg < 1) {
      printf("N (# U pts) must be positive!\n");
      usage();
      return 1;
    }
  }
  if (argc > 4) {
    sscanf(argv[4], "%d", &sort);
    if ((sort != 0) && (sort != 1) && (sort != 2)) {
      printf("sort must be 0, 1 or 2!\n");
      usage();
      return 1;
    }
  }
  if (argc > 5) {
    if (sscanf(argv[5], "%lf", &w) != 1) {
      usage();
      return 1;
    }
    upsampfac = (FLT)w;
    if (upsampfac <= (FLT)1.0) {
      printf("upsampfac must be >1.0!\n");
      usage();
      return 1;
    }
  }

  BIGINT N  = (BIGINT)round(pow(roughNg, 1.0 / d)); // grid size per dim
  BIGINT Ng = (BIGINT)pow(N, d);                    // total U grid size

  // allocate coordinate arrays
  std::vector<FLT> kx(M), ky(1), kz(1);
  if (d > 1) ky.resize(M);
  if (d > 2) kz.resize(M);

  // use complex arrays for NU strengths and uniform grid values
  // (here d_nonuniform stay fixed, while d_nuout is for dir=2 output only)
  std::vector<CPX> d_nonuniform(M), d_nuout(M), d_uniform(Ng);

  printf("spreadtestndall %dD: M=%.3g NU pts, Ng=%.3g U pts.\n", d, (double)M,
         (double)Ng);
  // random NU locs and strengths...
  printf("making random data, to be used at all tolerances...\n");
#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(dynamic, 100000)
    for (BIGINT i = 0; i < M; ++i) {
      kx[i] = randm11r(&se) * 3 * PI; // here [-3pi,3pi] tests folding, I guess
      if (d > 1) ky[i] = randm11r(&se) * 3 * PI;
      if (d > 2) kz[i] = randm11r(&se) * 3 * PI;
      d_nonuniform[i] = crandm11r(&se);
    }
  }

  std::vector<FLT> kxs(1), kys(1), kzs(1);    // set up tiny M=1 data to get kernel sum
  kxs[0] = kys[0] = kzs[0] = 0.0;  // 1 NU point loc, at nf/2 in each dim, on grid.
  std::vector<CPX> d_nus   = {CPX(1.0, 0.0)}; // with unit strength

  // where to stop (whether that causes warnings will depend on upsampfac):
  const auto max_digits = std::is_same_v<FLT, double> ? 15 : 7;

  for (int digits = 0; digits <= max_digits; ++digits) { // -------- loop # digits
    const auto tol = (FLT)pow(10.0, -digits);            // usual meaning of digits

    // set finufft options from CLI choices
    finufft_opts fopts;
    FINUFFT_DEFAULT_OPTS(&fopts);
    fopts.upsampfac          = upsampfac;
    fopts.spread_sort        = sort;
    fopts.debug              = 0; // hardwired for now
    fopts.spread_debug       = 0; // "
    fopts.showwarn           = 1;
    fopts.spreadinterponly   = 1; // key to use via FINUFFT API

    FINUFFT_PLAN plan{};
    BIGINT nmodes[3] = {N, N, N};
    int type         = 1; // in FINUFFT API this gives spread via execute
    int ier          = FINUFFT_MAKEPLAN(type, d, nmodes, +1, 1, tol, &plan, &fopts);
    if (ier > 1) {
      printf("error when creating the plan (ier=%d)!\n", ier);
      return ier;
    }

    // first use this plan for a M=1 tiny dir=1 problem to get the kernel sum...
    ier = FINUFFT_SETPTS(plan, 1, kxs.data(), kys.data(), kzs.data(), 0, nullptr, nullptr,
                         nullptr);
    if (ier != 0) {
      printf("error when setting NU pts (ier=%d)!\n", ier);
      FINUFFT_DESTROY(plan);
      return ier;
    }
    ier = FINUFFT_EXECUTE(plan, d_nus.data(), d_uniform.data());
    if (ier != 0) {
      printf("error when spreading M=1 pt for sum acc check (ier=%d)!\n", ier);
      FINUFFT_DESTROY(plan);
      return ier;
    }
    auto kersum = std::accumulate(d_uniform.begin(), d_uniform.end(), CPX(0.0, 0.0));

    // timing and call spread/interp through FINUFFT execute
    CNTime timer{};
    timer.start();
    ier = FINUFFT_SETPTS(plan, M, kx.data(), ky.data(), kz.data(), 0, nullptr, nullptr,
                         nullptr);
    if (ier != 0) {
      printf("error when setting NU pts (ier=%d)!\n", ier);
      FINUFFT_DESTROY(plan);
      return ier;
    }
    double t = timer.elapsedsec();
    printf("tol=%.3g:\tsetpts in %.3g s\n", (double)tol, t);

    // dir == 1: spread NU->U via execute ...................................
    timer.restart();
    ier = FINUFFT_EXECUTE(plan, d_nonuniform.data(), d_uniform.data()); // spread
    t   = timer.elapsedsec();
    if (ier != 0) {
      printf("exec error (ier=%d)!\n", ier);
      FINUFFT_DESTROY(plan);
      return ier;
    }
    printf("  dir=1 exec in %.3g s,   \t%.3g NU pt/s   ", t, M / t);
    // compare grid sum to predicted kersum times sum of strengths...
    CPX csum   = std::accumulate(d_nonuniform.begin(), d_nonuniform.end(), CPX(0.0, 0.0));
    CPX mass   = std::accumulate(d_uniform.begin(), d_uniform.end(), CPX(0.0, 0.0));
    FLT relerr = std::abs(mass - kersum * csum) / std::abs(kersum * csum);
    printf("\trel err %.3g\n", relerr);

    for (BIGINT i = 0; i < Ng; ++i) d_uniform[i] = CPX(1.0, 0.0); // const on grid

    // dir == 2: interpolate U->NU via execute_adjoint ........................
    timer.restart();
    ier = FINUFFT_EXECUTE_ADJOINT(plan, d_nuout.data(), d_uniform.data());
    t   = timer.elapsedsec();
    if (ier != 0) {
      printf("exec error (ier=%d)!\n", ier);
      FINUFFT_DESTROY(plan);
      return ier;
    }
    printf("  dir=2 exec in %.3g s,  \t%.3g NU pt/s   ", t, M / t);
    // interp-only test: compute sup error at NU points vs kersum
    FLT superr = 0.0;
    for (auto &cj : d_nuout) superr = std::max(superr, std::abs(cj - kersum));
    FLT relsuperr = superr / std::abs(kersum);
    printf("\trel err %.3g\n", relsuperr);

    FINUFFT_DESTROY(plan);
  } //                                          ---------------- end digits loop

  return 0;
}
