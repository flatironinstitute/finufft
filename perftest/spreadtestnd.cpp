/* Test executable for the 1D, 2D, or 3D C++ spreader, both directions,
   via the FINUFFT API (using opts.spreadinterponly=1).
   It checks speed, and basic correctness via the grid sum of the result.
   See usage() for usage.

 * Magland; expanded by Barnett 1/14/17. Better cmd line args 3/13/17
 * indep setting N 3/27/17. parallel rand() & sort flag 3/28/17
 * timing_flags 6/14/17. debug control 2/8/18. sort=2 opt 3/5/18, pad 4/24/18.
 * ier=1 warning not error, upsampfac 6/14/20.
 * Barbone, removed pirange 05/09/24.
 * Barbone switched to public FINUFFT API 11/07/2025
 * Barbone (Dec/25): Warn when deprecated CLI spreader knobs are requested.
 * Barnett 1/12/26: update for v2.5.0, removing all deprecated flags/args
*/

#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include <finufft/test_defs.h>

using namespace std;
using namespace finufft::utils;

/* clang-format off */
static void usage() {
  printf(
    "usage: spreadtestnd dims [M N [tol [sort [spread_debug [kerformula [upsampfac]]]]]]\n"
    "\twhere dims=1,2 or 3\n"
    "\tM=# nonuniform pts\n"
    "\tN=# uniform pts (rough total; per-dim N=round(N^(1/d)))\n"
    "\ttol=requested accuracy\n"
    "\tsort=0 (no), 1 (yes), 2 (auto; default)\n"
    "\tspread_debug=0,1,...\n"
    "\tspread_kerformula=0,1,... spread kernel type (>0 only for devs)\n"
    "\tupsampfac>1.0: sigma upsampling factor (typ range 1.2 to 2.5)\n"
    "\nexample: ./spreadtestnd 3 8e6 8e6 1e-6 2 0 0 1.5\n");
}
/* clang-format on */

int main(int argc, char *argv[]) {
  int d = 3;
  double w;
  double tol      = 1e-6;
  BIGINT M        = 1e6;
  BIGINT roughNg  = 1e6;
  int sort        = 2;
  int spread_debug= 0;
  int kerformula  = 0;
  FLT upsampfac   = 2.0;

  if (argc < 2 || argc == 3 || argc > 9) {
    usage();
    return (argc > 1);
  }
  if (sscanf(argv[1], "%d", &d) != 1 || d < 1 || d > 3) {
    printf("d must be 1, 2 or 3!\n");
    usage();
    return 1;
  }
  if (argc > 2) {
    if (sscanf(argv[2], "%lf", &w) != 1) {
      usage();
      return 1;
    }
    M = (BIGINT)w; // supports inputs like "1e6"
    if (M < 1) {
      printf("M (# NU pts) must be positive!\n");
      usage();
      return 1;
    }
    if (sscanf(argv[3], "%lf", &w) != 1) {
      usage();
      return 1;
    }
    roughNg = (BIGINT)w;
    if (roughNg < 1) {
      printf("N (# U pts) must be positive!\n");
      usage();
      return 1;
    }
  }
  if (argc > 4) {
    if (sscanf(argv[4], "%lf", &tol) != 1) {
      usage();
      return 1;
    }
  }
  if (argc > 5) {
    if (sscanf(argv[5], "%d", &sort) != 1 || (sort != 0 && sort != 1 && sort != 2)) {
      printf("sort must be 0, 1 or 2!\n");
      usage();
      return 1;
    }
  }
  if (argc > 6) {
    if (sscanf(argv[6], "%d", &spread_debug) != 1) {
      usage();
      return 1;
    }
  }
  if (argc > 7) {
    if (sscanf(argv[7], "%d", &kerformula) != 1) {
      usage();
      return 1;
    }
  }
  if (argc > 8) {
    if (sscanf(argv[8], "%lf", &w) != 1) {
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

  // Derive per-dim size and total U-grid size
  const auto N  = (BIGINT)llround(pow((long double)roughNg, 1.0L / (long double)d));
  const auto Ng = (BIGINT)pow((long double)N, (long double)d);

  // Allocate NU coords and data
  vector<FLT> kx(M), ky(d > 1 ? M : 1), kz(d > 2 ? M : 1);
  vector<CPX> d_nonuniform(M), d_uniform(Ng);

  // Options
  FINUFFT_PLAN plan{};
  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.upsampfac         = upsampfac;
  opts.spread_kerformula = kerformula;
  opts.spread_debug      = spread_debug;
  opts.spread_sort       = sort;
  opts.showwarn          = 1;
  opts.spreadinterponly  = 1; // the key to how this tester can use FINUFFT API

  BIGINT nmodes[3] = {N, N, N};

  // Make plan for type-1 (NU -> U). With spread/interp only, we can use adjoint too.
  int ier = FINUFFT_MAKEPLAN(1, d, nmodes, 1, 1, tol, &plan, &opts);
  if (ier > 1) {
    printf("error when creating the plan (ier=%d)!\n", ier);
    return ier;
  }

  // Initialize NU points (will be overwritten later), and set in plan
  ier = FINUFFT_SETPTS(plan, M, kx.data(), ky.data(), kz.data(), 0, nullptr, nullptr,
                       nullptr);
  if (ier != 0) {
    printf("error when setting NU pts (ier=%d)!\n", ier);
    FINUFFT_DESTROY(plan);
    return ier;
  }

  // Reference: spread a single source at origin to get kernel sum over grid
  d_nonuniform.assign(M, CPX(0.0, 0.0));
  d_nonuniform[0] = CPX(1.0, 0.0);
  kx[0]           = 0.0;
  if (d > 1) ky[0] = 0.0;
  if (d > 2) kz[0] = 0.0;

  ier = FINUFFT_EXECUTE(plan, d_nonuniform.data(), d_uniform.data());
  if (ier != 0) {
    printf("error when spreading M=1 pt for ref acc check (ier=%d)!\n", ier);
    FINUFFT_DESTROY(plan);
    return ier;
  }
  const CPX kersum = std::accumulate(d_uniform.begin(), d_uniform.end(), CPX(0.0, 0.0));

  // -------- Type-1 test (spread) --------
  printf("config: d=%d, per-dim Nj=%lld, total N=%.3g, M=%.3g, tol=%.3g\n", d,
         (long long)N, (double)Ng, (double)M, tol);
  printf("        sort=%d, spread_debug=%d, kerform=%d, upsamp=%.3g\n", sort,
         spread_debug, kerformula, (double)upsampfac);

  printf("making random data for dir=1...\n");
#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(static)
    for (BIGINT i = 0; i < M; ++i) {
      // Stress periodic wrapping: coordinates in [-3π, 3π]
      kx[i] = randm11r(&se) * (FLT)(3 * PI);
      if (d > 1) ky[i] = randm11r(&se) * (FLT)(3 * PI);
      if (d > 2) kz[i] = randm11r(&se) * (FLT)(3 * PI);
      d_nonuniform[i] = crandm11r(&se);
    }
  }

  CNTime timer{};
  timer.start();
  ier = FINUFFT_SETPTS(plan, M, kx.data(), ky.data(), kz.data(), 0, nullptr, nullptr,
                       nullptr);
  ier = FINUFFT_EXECUTE(plan, d_nonuniform.data(), d_uniform.data());
  const double t1 = timer.elapsedsec();
  if (ier != 0) {
    printf("error in dir=1 execute (ier=%d)!\n", ier);
    FINUFFT_DESTROY(plan);
    return ier;
  } else
    printf("\tsetpts+exec done in %.3g s", t1);

  // Compute total input strength and total output mass, compare with kersum
  CPX csum   = std::accumulate(d_nonuniform.begin(), d_nonuniform.end(), CPX(0.0, 0.0));
  CPX mass   = std::accumulate(d_uniform.begin(), d_uniform.end(), CPX(0.0, 0.0));
  FLT relerr = std::abs(mass - kersum * csum) / std::abs(mass);
  printf("\t\trel err in grid total: %.3g\n", relerr);

  // -------- Type-2 test (U -> NU) using a separate type-2 plan --------
  printf("making random NU pts for dir=2...\n");
  // sets uniform grid to 1.0 everywhere...
  std::fill(d_uniform.begin(), d_uniform.end(), CPX(1.0, 0.0));
#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(static)
    for (BIGINT i = 0; i < M; ++i) {
      kx[i] = randm11r(&se) * (FLT)(3 * PI);
      if (d > 1) ky[i] = randm11r(&se) * (FLT)(3 * PI);
      if (d > 2) kz[i] = randm11r(&se) * (FLT)(3 * PI);
    }
  }
  // Create a separate plan for type-2 (U -> NU) and use it instead of the
  // adjoint call on the type-1 plan. This matches the public API usage where
  // a type-2 plan is created and executed.
  FINUFFT_PLAN plan2{};
  ier = FINUFFT_MAKEPLAN(2, d, nmodes, 1, 1, tol, &plan2, &opts);
  if (ier > 1) {
    printf("error when creating the type-2 plan (ier=%d)!\n", ier);
    FINUFFT_DESTROY(plan);
    return ier;
  }

  timer.restart();
  ier = FINUFFT_SETPTS(plan2, M, kx.data(), ky.data(), kz.data(), 0, nullptr, nullptr,
                       nullptr);
  if (ier != 0) {
    printf("error when setting NU pts for type-2 plan (ier=%d)!\n", ier);
    FINUFFT_DESTROY(plan);
    FINUFFT_DESTROY(plan2);
    return ier;
  }

  ier = FINUFFT_EXECUTE(plan2, d_nonuniform.data(), d_uniform.data());
  const double t2 = timer.elapsedsec();
  if (ier != 0) {
    printf("error in dir=2 execute (ier=%d)!\n", ier);
    FINUFFT_DESTROY(plan);
    FINUFFT_DESTROY(plan2);
    return ier;
  } else
    printf("\tsetpts+exec done in %.3g s", t2);

  // interp-only check: since grid=1.0 was done above, interp should give kersum
  // (const) at all NU points in d_nonuniform. Compute sup error vs kersum:
  FLT superr = 0.0;
  for (auto &cj : d_nonuniform) superr = std::max(superr, std::abs(cj - kersum));
  FLT relsuperr = superr / std::abs(kersum);
  printf("\t\tmax rel err vs const:  %.3g\n", relsuperr);
  FINUFFT_DESTROY(plan);
  FINUFFT_DESTROY(plan2);
  return 0;
}
