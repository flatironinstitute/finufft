/* Test executable for the 1D, 2D, or 3D spreader, both directions.
 * This version uses the public FINUFFT guru API to exercise the spreader
 * behavior and checks correctness via kernel sums.

 * Barbone, using the public finufft API 11/07/2025.
 * Barnett over-simplified to remove misleading opts, 1/12/26.
 Todo: update args + screen output like spreadtestnd.cpp
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
      "usage: spreadtestnd dims [M N [dir [sort]]]\n"
      "\twhere dims=1,2 or 3\n"
      "\tM=# nonuniform pts\n"
      "\tN=# uniform pts (total)\n"
      "\tdir=direction (1=spread, 2=interpolate)\n"
      "\tsort=0 (don't sort NU pts), 1 (do), or 2 (maybe sort; default)\n"
      "\n"
      "example: ./spreadtestndall 1 1e6 1e6 1 2\n");
}
/* clang-format on */

int main(int argc, char *argv[]) {
  int d;
  double w;
  // Cmd line args & their defaults...
  int dir         = 1;   // default: spread (NU->U)
  BIGINT M        = 1e6; // default # NU pts
  BIGINT roughNg  = 1e6; // default # U pts
  int sort        = 2;   // spread_sort
  int debug       = 0;
  FLT upsampfac   = 2.0;

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
  if (argc > 4) sscanf(argv[4], "%d", &dir);
  if (argc > 5) {
    sscanf(argv[5], "%d", &sort);
    if ((sort != 0) && (sort != 1) && (sort != 2)) {
      printf("sort must be 0, 1 or 2!\n");
      usage();
      return 1;
    }
  }

  BIGINT N  = (BIGINT)round(pow(roughNg, 1.0 / d)); // grid size per dim
  BIGINT Ng = (BIGINT)pow(N, d);                    // total grid points

  // allocate coordinate arrays
  std::vector<FLT> kx(M), ky(1), kz(1);
  if (d > 1) ky.resize(M);
  if (d > 2) kz.resize(M);

  // use complex arrays for NU strengths and uniform grid values
  std::vector<CPX> d_nonuniform(M), d_uniform(Ng);

  const auto max_digits = std::is_same_v<FLT, double> ? 17 : 9;

  for (int digits = 2; digits < max_digits; ++digits) {
    const auto tol = (FLT)(10.0 * pow(10.0, -digits));
    printf("digits=%d, tol = %.3g\n", digits, (double)tol);

    // set finufft options from CLI choices
    finufft_opts fopts;
    FINUFFT_DEFAULT_OPTS(&fopts);
    fopts.upsampfac          = upsampfac;
    fopts.spread_sort        = sort;
    fopts.debug              = debug;
    fopts.showwarn           = 1;
    fopts.spreadinterponly   = 1; // key to use via FINUFFT API

    FINUFFT_PLAN plan{};
    BIGINT nmodes[3] = {N, N, N};
    int ier          = FINUFFT_MAKEPLAN(dir, d, nmodes, +1, 1, tol, &plan, &fopts);
    if (ier > 1) {
      printf("error when creating the plan (ier=%d)!\n", ier);
      return ier;
    }

    ier = FINUFFT_SETPTS(plan, M, kx.data(), ky.data(), kz.data(), 0, nullptr, nullptr,
                         nullptr);
    if (ier != 0) {
      printf("error when setting NU pts (ier=%d)!\n", ier);
      FINUFFT_DESTROY(plan);
      return ier;
    }

    d_nonuniform.assign(M, CPX(0.0, 0.0));
    d_uniform.assign(Ng, CPX(0.0, 0.0));
    kx[0] = ky[0] = kz[0] = (FLT)PI / 2.0; // NU point location (unchanged)
    auto kersum           = CPX(0.0, 0.0);
    if (dir == 1) {
      // type-1 (NU -> U): place unit at NU point, spread to grid and sum grid
      d_nonuniform[0] = CPX(1.0, 0.0);
      ier             = FINUFFT_EXECUTE(plan, d_nonuniform.data(), d_uniform.data());
      if (ier != 0) {
        printf("error when spreading M=1 pt for ref acc check (ier=%d)!\n", ier);
        FINUFFT_DESTROY(plan);
        return ier;
      }
      kersum = std::accumulate(d_uniform.begin(), d_uniform.end(), CPX(0.0, 0.0));
    } else {
      // type-2 (U -> NU): set uniform grid to ones, interpolate to NU point
      for (BIGINT i = 0; i < Ng; ++i) d_uniform[i] = CPX(1.0, 0.0);
      ier = FINUFFT_EXECUTE(plan, d_nonuniform.data(), d_uniform.data());
      if (ier != 0) {
        printf("error when interpolating for ref acc check (ier=%d)!\n", ier);
        FINUFFT_DESTROY(plan);
        return ier;
      }
      // result at the single NU point is the kernel sum
      kersum = d_nonuniform[0];
    }

    // now random data for large test
    printf("making random data...\n");
#pragma omp parallel
    {
      unsigned int se = MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(dynamic, 1000000)
      for (BIGINT i = 0; i < M; ++i) {
        kx[i] = randm11r(&se) * 3 * PI;
        if (d > 1) ky[i] = randm11r(&se) * 3 * PI;
        if (d > 2) kz[i] = randm11r(&se) * 3 * PI;
        d_nonuniform[i] = crandm11r(&se);
      }
    }

    // timing and call spread/interp through FINUFFT execute
    CNTime timer{};
    double t;

    if (dir == 1) {
      printf("spreadinterp %dD, %.3g U pts, dir=%d, tol=%.3g:\n", d, (double)Ng, 1,
             (double)tol);
      timer.start();
      ier = FINUFFT_EXECUTE(plan, d_nonuniform.data(), d_uniform.data());
      t   = timer.elapsedsec();
      if (ier != 0) {
        printf("error (ier=%d)!\n", ier);
        FINUFFT_DESTROY(plan);
        return ier;
      }
      printf("\t%.3g NU pts in %.3g s \t%.3g pts/s\n", (double)M, t, M / t);
      // compare grid sum to predicted kersum*sum(c)
      CPX csum = std::accumulate(d_nonuniform.begin(), d_nonuniform.end(), CPX(0.0, 0.0));
      CPX mass = std::accumulate(d_uniform.begin(), d_uniform.end(), CPX(0.0, 0.0));
      FLT relerr = std::abs(mass - kersum * csum) / std::abs(mass);
      printf("\trel err in total over grid: %.3g\n", relerr);

    } else { // dir == 2: interpolate U->NU via execute_adjoint
      printf("making more random NU pts...\n");
      for (BIGINT i = 0; i < Ng; ++i) d_uniform[i] = CPX(1.0, 0.0);

      printf("spreadinterpall %dD, %.3g U pts, dir=%d, tol=%.3g:\n", d, (double)Ng, 2,
             (double)tol);
      timer.restart();
      ier = FINUFFT_EXECUTE(plan, d_nonuniform.data(), d_uniform.data());
      t   = timer.elapsedsec();
      if (ier != 0) {
        printf("error (ier=%d)!\n", ier);
        FINUFFT_DESTROY(plan);
        return ier;
      }
      printf("\t%.3g NU pts in %.3g s \t%.3g pts/s\n", (double)M, t, M / t);
      // interp-only test: compute sup error at NU points vs kersum
      FLT superr = 0.0;
      for (auto &cj : d_nonuniform) superr = std::max(superr, std::abs(cj - kersum));
      FLT relsuperr = superr / std::abs(kersum);
      printf("\trel sup err %.3g\n", relsuperr);
    }

    FINUFFT_DESTROY(plan);
  }

  return 0;
}
