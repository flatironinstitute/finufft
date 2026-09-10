/* test/cuda/cutolsweep: pass-fail accuracy test for cuFINUFFT that sweeps the full
   range of tolerances, dims, types, for a set of upsampfacs.
   The GPU counterpart of test/tolsweep.cpp: same problem sizes, same tol ladder,
   same slack factors and error floors, so the two libraries are held to one
   standard. Uses relative L2 error norms against a direct reference evaluation on
   the host.
   Exit code: zero if success, nonzero upon failure.

   Barbone 8/26, after Barnett's test/tolsweep.cpp.
*/

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <cufinufft.h>
#include <finufft_common/common.h>

#include <thrust/device_vector.h>

// test utilities: direct DFT and norm helpers (shared with the CPU tolsweep)
#include "../utils/dirft1d.hpp"
#include "../utils/dirft2d.hpp"
#include "../utils/dirft3d.hpp"
#include "../utils/norms.hpp"

// One cuFINUFFT transform, host in / host out. Returns the makeplan/setpts error
// code, or 0. On type 2 the answer lands in c, otherwise in F.
template<typename T>
int run_one(int type, int dim, const int64_t *Nm, int isign, double tol,
            const cufinufft_opts &opts, int M, const std::vector<T> &x,
            const std::vector<T> &y, const std::vector<T> &z, int N,
            const std::vector<T> &X, const std::vector<T> &Y, const std::vector<T> &Z,
            std::vector<std::complex<T>> &c, std::vector<std::complex<T>> &F) {
  using Plan =
      std::conditional_t<std::is_same_v<T, float>, cufinufftf_plan, cufinufft_plan>;
  using Cpx =
      std::conditional_t<std::is_same_v<T, float>, cuFloatComplex, cuDoubleComplex>;
  thrust::device_vector<T> d_x(x), d_y(y), d_z(z), d_X(X), d_Y(Y), d_Z(Z);
  thrust::device_vector<std::complex<T>> d_c(c), d_F(F);

  Plan plan;
  cufinufft_opts o = opts; // makeplan may overwrite auto-choices
  int ier;
  if constexpr (std::is_same_v<T, float>)
    ier = cufinufftf_makeplan(type, dim, Nm, isign, 1, T(tol), &plan, &o);
  else
    ier = cufinufft_makeplan(type, dim, Nm, isign, 1, T(tol), &plan, &o);
  if (ier) return ier;

  auto *px = d_x.data().get(), *py = dim > 1 ? d_y.data().get() : nullptr,
       *pz = dim > 2 ? d_z.data().get() : nullptr;
  auto *pX = d_X.data().get(), *pY = dim > 1 ? d_Y.data().get() : nullptr,
       *pZ = dim > 2 ? d_Z.data().get() : nullptr;
  auto *pc = (Cpx *)d_c.data().get();
  auto *pF = (Cpx *)d_F.data().get();
  if constexpr (std::is_same_v<T, float>) {
    ier = cufinufftf_setpts(plan, M, px, py, pz, N, pX, pY, pZ);
    if (!ier) ier = cufinufftf_execute(plan, pc, pF);
    cufinufftf_destroy(plan);
  } else {
    ier = cufinufft_setpts(plan, M, px, py, pz, N, pX, pY, pZ);
    if (!ier) ier = cufinufft_execute(plan, pc, pF);
    cufinufft_destroy(plan);
  }
  if (ier) return ier;

  if (type == 2)
    thrust::copy(d_c.begin(), d_c.end(), c.begin());
  else
    thrust::copy(d_F.begin(), d_F.end(), F.begin());
  return 0;
}

template<typename T> int sweep(int verbose, int debug) {
  using Cpx = std::complex<T>;
  using finufft::common::PI;

  // Problem sizes, tol ladder and slack factors: identical to test/tolsweep.cpp.
  const int M                 = 500; // # sources (balance runtime vs rand-averaging)
  // N vectors to test: first triplet is for dim=1, then for dim=2, etc...
  int64_t Nm_alldims[3][3]    = {{50, 1, 1}, {25, 40, 1}, {10, 11, 12}};
  const int isign             = +1;
  const double tolslack[3]    = {4.0, 4.0, 5.0}; // slack parameter for each type
  const double tolsperdecade  = 4;               // controls effort (tol resolution)
  const double tolstep        = std::pow(10.0, -1.0 / tolsperdecade);
  constexpr T EPSILON         = std::numeric_limits<T>::epsilon();
  const double mintol         = 0.5 * EPSILON; // where to stop (catch warns)
  const int ntols             = std::ceil(std::log(mintol) / std::log(tolstep));

  // test set of upsampfacs each with matching error floor for each dim...
  const int nu                = 2;
  const double upsampfac[nu]  = {1.25, 2.0};
  const double floor_f[nu][3] = {{1e-4, 1e-4, 2e-4}, {2e-5, 2e-5, 1e-5}};
  const double floor_d[nu][3] = {{1e-9, 2e-9, 3e-8}, {3e-14, 3e-14, 3e-14}};
  const bool singleprec       = std::is_same_v<T, float>;

  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.debug = debug;

  std::vector<T> x(M), y(M), z(M), X, Y, Z;
  std::vector<Cpx> c(M), ce(M), F, Fe;
  std::default_random_engine eng(42);  // fix seed
  std::uniform_real_distribution<T> m11(-1, 1), z01(0, 1);
  int nfailtot = 0;                    // overall count across all dims, USF, tols, types

  for (int dim = 1; dim <= 3; ++dim) { /////////////////////// loop over dims
    if (verbose) printf("cutolsweep: %dD =============================\n", dim);
    int64_t *Nm = Nm_alldims[dim - 1];
    int64_t N   = Nm[0] * Nm[1] * Nm[2]; // tot # modes, or freq-pts for type 3
    X.assign(N, 0);
    Y.assign(N, 0);
    Z.assign(N, 0);
    F.assign(N, 0);
    Fe.assign(N, 0);

    for (int u = 0; u < nu; ++u) { // ===================== loop over upsampfacs
      opts.upsampfac = upsampfac[u];
      if (verbose) printf(" upsampfac = %.3g -----------------\n", opts.upsampfac);

      double worstfac[3] = {0};                 // largest clearance for each type
      double tol         = 1.0;                 // starting (max) tol to test
      int npass[3] = {0}, nfail[3] = {0};       // counts for each type
      for (int t = 0; t < ntols; ++t) {         // ............... loop over tols
        for (int type = 1; type <= 3; ++type) { // ------------- loop over types

          // fresh data each test & type, even data not needed for that type
          for (int j = 0; j < M; ++j) {
            x[j] = T(PI) * m11(eng);
            y[j] = T(PI) * m11(eng);
            z[j] = T(PI) * m11(eng);
            c[j] = Cpx(m11(eng), m11(eng));
          }
          for (int64_t k = 0; k < N; ++k) {
            X[k] = T(Nm[0]) * z01(eng); // type 3: scale freq NU pts by "mode" sizes
            Y[k] = T(Nm[1]) * z01(eng);
            Z[k] = T(Nm[2]) * z01(eng);
            F[k] = Cpx(m11(eng), m11(eng));
          }
          const int ier = run_one<T>(type, dim, Nm, isign, tol, opts, M, x, y, z, int(N),
                                     X, Y, Z, c, F);
          if (ier) {
            fprintf(stderr, "   cutolsweep: %dD%d tol=%.3g failed! ier=%d\n", dim, type,
                    tol, ier);
            return 1;
          }

          if (dim == 1) // the relevant one of nine direct "exact" evals...
            if (type == 1)
              dirft1d1<int64_t>(int64_t(M), x, c, isign, Nm[0], Fe);
            else if (type == 2)
              dirft1d2<int64_t>(int64_t(M), x, ce, isign, Nm[0], F);
            else
              dirft1d3<int64_t>(int64_t(M), x, c, isign, Nm[0], X, Fe);
          else if (dim == 2)
            if (type == 1)
              dirft2d1<int64_t>(int64_t(M), x, y, c, isign, Nm[0], Nm[1], Fe);
            else if (type == 2)
              dirft2d2<int64_t>(int64_t(M), x, y, ce, isign, Nm[0], Nm[1], F);
            else
              dirft2d3<int64_t>(int64_t(M), x, y, c, isign, N, X, Y, Fe);
          else // dim=3
            if (type == 1)
              dirft3d1<int64_t>(int64_t(M), x, y, z, c, isign, Nm[0], Nm[1], Nm[2], Fe);
            else if (type == 2)
              dirft3d2<int64_t>(int64_t(M), x, y, z, ce, isign, Nm[0], Nm[1], Nm[2], F);
            else
              dirft3d3<int64_t>(int64_t(M), x, y, z, c, isign, N, X, Y, Z, Fe);

          double relerr; // the relevant error metric
          if (type == 2)
            relerr = relerrtwonorm<int64_t>(int64_t(M), ce, c); // ce comes 1st
          else
            relerr = relerrtwonorm<int64_t>(N, Fe, F);

          const int ti          = type - 1; // index for 3-el arrays
          const double flr      = singleprec ? floor_f[u][dim - 1] : floor_d[u][dim - 1];
          const double req      = std::max(flr, tolslack[ti] * tol); // threshold
          const double clearfac = relerr / req; // factor beating req (<=1 ok)
          worstfac[ti]          = std::max(worstfac[ti], clearfac);
          if (relerr <= req) {                  // note relerr=NaN will not pass
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
          }
        } // -----------------------------

        tol *= tolstep; // reduce tol in geometric progression
      } // ...........................................

      if (verbose)
        for (int ti = 0; ti < 3; ++ti)
          printf("  %dD%d summary: %d pass, %d fail. worstfac=%g\n", dim, ti + 1,
                 npass[ti], nfail[ti], worstfac[ti]);
      nfailtot += nfail[0] + nfail[1] + nfail[2];
    } // ==========================
  } //////////////////////////

  return nfailtot > 0;
}

int main(int argc, char *argv[]) {
  std::string prec = "d";
  int verbose = 1, debug = 0;
  for (int ai = 1; ai < argc; ++ai)
    if (std::string(argv[ai]) == "-h" || std::string(argv[ai]) == "--help") {
      printf("Usage: %s [prec [verbose [debug]]]\n", argv[0]);
      printf("  prec          : f (single) or d (double, default)\n");
      printf("  verbose       : 0 silent, 1 summary (default), 3 every case\n");
      printf("  debug         : passed to opts.debug for every call\n");
      return 0;
    }
  if (argc > 1) prec = argv[1];
  if (argc > 2) verbose = std::atoi(argv[2]);
  if (argc > 3) debug = std::atoi(argv[3]);

  if (prec == "f") return sweep<float>(verbose, debug);
  if (prec == "d") return sweep<double>(verbose, debug);
  fprintf(stderr, "%s: prec must be f or d, got %s\n", argv[0], prec.c_str());
  return 1;
}
