#include <finufft/test_defs.h>
// this enforces recompilation, responding to SINGLE...
#include "directft/dirft3d.cpp"
using namespace std;
using namespace finufft::utils;

const char *help[] = {
    "Test spread_kerevalmeth=0 & 1 match, for 3 types of 3D transf, either prec.",
    "Usage: finufft3dkernel_test Nmodes1 Nmodes2 Nmodes3 Nsrc",
    "\t[tol] error tolerance (default 1e-6)",
    "\t[debug] (default 0) 0: silent, 1: text, 2: as 1 but also spreader",
    "\t[spread_sort] (default 2) 0: don't sort NU pts, 1: do, 2: auto",
    "\t[upsampfac] (default 2.0)",
    "\teg: finufft3dkernel_test 100 200 50 1e6 1e-12 0 2 0.0",
    "\tnotes: exit code 1 if any error > tol",
    nullptr};
/**
 * @brief Test the 3D NUFFT of type 1, 2, and 3.
 * It evaluates the error of the kernel evaluation methods.
 * It uses err(a,b)=||a-b||_2 / ||a||_2 as the error metric.
 * It return FINUFFT error code if it is not 0.
 * It returns 1 if any error exceeds tol.
 * It returns 0 if test passes.
 */
int main(int argc, char *argv[]) {
  BIGINT M, N1, N2, N3;      // M = # srcs, N1,N2,N3 = # modes
  double w, tol      = 1e-6; // default
  double err, errmax = 0;
  finufft_opts opts0, opts1;
  FINUFFT_DEFAULT_OPTS(&opts0);
  FINUFFT_DEFAULT_OPTS(&opts1);
  opts0.spread_kerevalmeth = 0;
  opts1.spread_kerevalmeth = 1;
  // opts.fftw = FFTW_MEASURE;  // change from usual FFTW_ESTIMATE
  // opts.spread_max_sp_size = 3e4; // override test
  // opts.spread_nthr_atomic = 15;  // "
  int isign = +1; // choose which exponential sign to test
  if (argc < 5 || argc > 10) {
    for (int i = 0; help[i]; ++i) fprintf(stderr, "%s\n", help[i]);
    return 2;
  }
  sscanf(argv[1], "%lf", &w);
  N1 = (BIGINT)w;
  sscanf(argv[2], "%lf", &w);
  N2 = (BIGINT)w;
  sscanf(argv[3], "%lf", &w);
  N3 = (BIGINT)w;
  sscanf(argv[4], "%lf", &w);
  M = (BIGINT)w;
  if (argc > 5) sscanf(argv[5], "%lf", &tol);
  if (argc > 6) sscanf(argv[6], "%d", &opts0.debug); // can be 0,1 or 2
  opts0.spread_debug = (opts0.debug > 1) ? 1 : 0;    // see output from spreader
  if (argc > 7) sscanf(argv[7], "%d", &opts0.spread_sort);
  if (argc > 8) {
    sscanf(argv[8], "%lf", &w);
    opts0.upsampfac = (FLT)w;
  }

  opts1                    = opts0;
  opts0.spread_kerevalmeth = 0;
  opts1.spread_kerevalmeth = 1;

  cout << scientific << setprecision(15);
  const BIGINT N = N1 * N2 * N3;

  std::vector<FLT> x(M);         // NU pts x coords
  std::vector<FLT> y(M);         // NU pts y coords
  std::vector<FLT> z(M);         // NU pts z coords
  std::vector<CPX> c0(M), c1(N); // strengths
  std::vector<CPX> F0(N);        // mode ampls kereval 0
  std::vector<CPX> F1(N);        // mode ampls kereval 1
#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM(); // needed for parallel random #s
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT j = 0; j < M; ++j) {
      x[j]  = M_PI * randm11r(&se);
      y[j]  = M_PI * randm11r(&se);
      z[j]  = M_PI * randm11r(&se);
      c0[j] = crandm11r(&se);
    }
  }
  c1 = c0;                     // copy strengths
  printf("test 3d type 1:\n"); // -------------- type 1
  printf("kerevalmeth 0:\n");
  CNTime timer{};
  timer.start();
  int ier = FINUFFT3D1(M, x.data(), y.data(), z.data(), c0.data(), isign, tol, N1, N2, N3,
                       F0.data(), &opts0);
  double ti = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("     %lld NU pts to (%lld,%lld,%lld) modes in %.3g s \t%.3g NU pts/s\n",
           (long long)M, (long long)N1, (long long)N2, (long long)N3, ti, M / ti);
  printf("kerevalmeth 1:\n");
  timer.restart();
  ier = FINUFFT3D1(M, x.data(), y.data(), z.data(), c0.data(), isign, tol, N1, N2, N3,
                   F1.data(), &opts1);
  ti  = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("     %lld NU pts to (%lld,%lld,%lld) modes in %.3g s \t%.3g NU pts/s\n",
           (long long)M, (long long)N1, (long long)N2, (long long)N3, ti, M / ti);

  err    = relerrtwonorm(N, F0.data(), F1.data());
  errmax = max(err, errmax);
  printf("\ttype 1 rel l2-err in F is %.3g\n", err);
  // copy F0 to F1 so that we can test type 2
  F1 = F0;
  printf("kerevalmeth 0:\n");
  timer.restart();
  ier = FINUFFT3D2(M, x.data(), y.data(), z.data(), c0.data(), isign, tol, N1, N2, N3,
                   F0.data(), &opts0);
  ti  = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("     (%lld,%lld,%lld) modes to %lld NU pts in %.3g s \t%.3g NU pts/s\n",
           (long long)N1, (long long)N2, (long long)N3, (long long)M, ti, M / ti);
  printf("kerevalmeth 1:\n");
  timer.restart();
  ier = FINUFFT3D2(M, x.data(), y.data(), z.data(), c1.data(), isign, tol, N1, N2, N3,
                   F0.data(), &opts1);
  ti  = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("     (%lld,%lld,%lld) modes to %lld NU pts in %.3g s \t%.3g NU pts/s\n",
           (long long)N1, (long long)N2, (long long)N3, (long long)M, ti, M / ti);
  err    = relerrtwonorm(M, c0.data(), c1.data());
  errmax = std::max(err, errmax);
  printf("\ttype 2 rel l2-err in c is %.3g\n", err);

  printf("test 3d type 3:\n"); // -------------- type 3
#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT j = 0; j < M; ++j) {
      x[j] = 2.0 + M_PI * randm11r(&se);  // new x_j srcs, offset from origin
      y[j] = -3.0 + M_PI * randm11r(&se); // " y_j
      z[j] = 1.0 + M_PI * randm11r(&se);  // " z_j
    }
  }
  std::vector<FLT> s(N); // targ freqs (1-cmpt)
  std::vector<FLT> t(N); // targ freqs (2-cmpt)
  std::vector<FLT> u(N); // targ freqs (3-cmpt)

  timer.restart();
  printf("kerevalmeth 0:\n");
  ier = FINUFFT3D3(M, x.data(), y.data(), z.data(), c0.data(), isign, tol, N, s.data(),
                   t.data(), u.data(), F0.data(), &opts0);
  ti  = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("\t%lld NU to %lld NU in %.3g s         \t%.3g tot NU pts/s\n", (long long)M,
           (long long)N, ti, (M + N) / ti);
  timer.restart();
  printf("kerevalmeth 1:\n");
  ier = FINUFFT3D3(M, x.data(), y.data(), z.data(), c0.data(), isign, tol, N, s.data(),
                   t.data(), u.data(), F1.data(), &opts1);
  ti  = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("\t%lld NU to %lld NU in %.3g s         \t%.3g tot NU pts/s\n", (long long)M,
           (long long)N, ti, (M + N) / ti);
  err    = relerrtwonorm(N, F0.data(), F1.data());
  errmax = max(err, errmax);
  printf("\ttype 3 rel l2-err in F is %.3g\n", err);
  // return 1 if any error exceeds tol
  // or return finufft error code if it is not 0
  return (errmax > tol);
}
