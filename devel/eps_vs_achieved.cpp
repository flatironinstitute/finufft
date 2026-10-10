/* devel/eps_vs_achieved: sweep requested tol, report achieved relative L2 error
   for FINUFFT types 1,2,3 at sigma {0=auto, 2.0}, dims 1..3, at tolsweep's sizes
   (M=500; N per dim {50, 25x40, 10x11x12}). Emits one CSV row per
   (prec,dim,type,sigma,tol). Pairs with devel/eps_vs_achieved.py for plotting.
   Built twice (double, and -DSINGLE). Same direct-reference error metric as
   test/tolsweep.cpp. opts.allow_eps_too_small=1 so the rounding floor shows.
*/
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <finufft/test_defs.hpp>
#include <vector>

#include "../test/utils/dirft1d.hpp"
#include "../test/utils/dirft2d.hpp"
#include "../test/utils/dirft3d.hpp"
#include "../test/utils/norms.hpp"

int main() {
  BIGINT M                = 500;
  BIGINT Nm_alldims[3][3] = {{50, 1, 1}, {25, 40, 1}, {10, 11, 12}};
  int ntr                 = 1;
  int isign               = +1;
  double tolsperdecade    = 8;
  double tolstep          = pow(10.0, -1.0 / tolsperdecade);
  constexpr FLT EPSILON   = std::numeric_limits<FLT>::epsilon();
  // user-requested tol ranges: double 1e-2..1e-13, float 1e-2..1e-6; keep a few
  // extra decades past the floor so the rounding-floor blowup is visible
  double tolmax           = 1e-2;
  double tolmin           = EPSILON / 100.0;
  int ntols               = std::ceil(log(tolmin / tolmax) / log(tolstep)) + 1;
  const double sigmas[2]  = {0.0, 2.0};

  finufft_opts opts{};
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.nthreads            = 1;
  opts.showwarn            = 0;
  opts.allow_eps_too_small = 1;

  std::vector<FLT> x(M), y(M), z(M), X, Y, Z;
  std::vector<CPX> c(M), ce(M), F, Fe, c0(M), F0;
  srand(42);

#ifdef SINGLE
  const char *precname = "float";
#else
  const char *precname = "double";
#endif
  printf("prec,dim,type,sigma,tol,relerr,ier\n");

  for (int dim = 1; dim <= 3; ++dim) {
    BIGINT *Nm = Nm_alldims[dim - 1];
    BIGINT N   = Nm[0] * Nm[1] * Nm[2];
    X.resize(N);
    Y.resize(N);
    Z.resize(N);
    F.resize(N);
    Fe.resize(N);

    for (double sigma : sigmas) {
      opts.upsampfac = sigma;
      // One fixed problem per (dim, sigma) so points along each curve are
      // comparable; inputs drawn once here, reused across tols and types.
      // c0/F0 are the pristine copies; c/F get clobbered by EXECUTE/dirft each
      // iteration and are restored below.
      for (BIGINT j = 0; j < M; ++j) {
        x[j]  = PI * randm11();
        y[j]  = PI * randm11();
        z[j]  = PI * randm11();
        c0[j] = crandm11();
      }
      F0.resize(N);
      for (BIGINT k = 0; k < N; ++k) {
        X[k]  = Nm[0] * rand01();
        Y[k]  = Nm[1] * rand01();
        Z[k]  = Nm[2] * rand01();
        F0[k] = crandm11();
      }
      double tol = tolmax;
      for (int t = 0; t < ntols; ++t) {
        for (int type = 1; type <= 3; ++type) {
          c                 = c0;
          F                 = F0;
          FINUFFT_PLAN plan = nullptr;
          int ier = FINUFFT_MAKEPLAN(type, dim, Nm, isign, ntr, (FLT)tol, &plan, &opts);
          int ier_set = FINUFFT_ERR_PLAN_NOTVALID, ier_ex = FINUFFT_ERR_PLAN_NOTVALID;
          if (ier == 0) {
            ier_set = FINUFFT_SETPTS(plan, M, x.data(), y.data(), z.data(), N, X.data(),
                                     Y.data(), Z.data());
            if (ier_set == 0) ier_ex = FINUFFT_EXECUTE(plan, c.data(), F.data());
            FINUFFT_DESTROY(plan);
          }
          const int ier_all = ier ? ier : (ier_set ? ier_set : ier_ex);
          // relerr valid only on success; NaN flags a failed row in the CSV.
          double relerr     = std::numeric_limits<double>::quiet_NaN();
          if (ier_all == 0) {
            // direct exact eval, matching tolsweep
            if (dim == 1) {
              if (type == 1)
                dirft1d1<BIGINT>(M, x, c, isign, Nm[0], Fe);
              else if (type == 2)
                dirft1d2<BIGINT>(M, x, ce, isign, Nm[0], F);
              else
                dirft1d3<BIGINT>(M, x, c, isign, Nm[0], X, Fe);
            } else if (dim == 2) {
              if (type == 1)
                dirft2d1<BIGINT>(M, x, y, c, isign, Nm[0], Nm[1], Fe);
              else if (type == 2)
                dirft2d2<BIGINT>(M, x, y, ce, isign, Nm[0], Nm[1], F);
              else
                dirft2d3<BIGINT>(M, x, y, c, isign, N, X, Y, Fe);
            } else {
              if (type == 1)
                dirft3d1<BIGINT>(M, x, y, z, c, isign, Nm[0], Nm[1], Nm[2], Fe);
              else if (type == 2)
                dirft3d2<BIGINT>(M, x, y, z, ce, isign, Nm[0], Nm[1], Nm[2], F);
              else
                dirft3d3<BIGINT>(M, x, y, z, c, isign, N, X, Y, Z, Fe);
            }
            if (type == 2)
              relerr = relerrtwonorm<BIGINT>(M, ce, c);
            else
              relerr = relerrtwonorm<BIGINT>(N, Fe, F);
          }

          printf("%s,%d,%d,%.3g,%.3e,%.3e,%d\n", precname, dim, type, sigma, tol, relerr,
                 ier_all);
        }
        tol *= tolstep;
      }
    }
  }
  return 0;
}
