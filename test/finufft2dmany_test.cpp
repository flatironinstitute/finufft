#include <finufft/test_defs.h>
// this enforces recompilation, responding to SINGLE...
#include "directft/dirft2d.cpp"
using namespace std;
using namespace finufft::utils;

const char *help[] = {
    "Tester for FINUFFT in 2d, vectorized, all 3 types, either precision.",
    "",
    "Usage: finufft2dmany_test ntrans Nmodes1 Nmodes2 Nsrc [tol [debug [spread_thread "
    "[maxbatchsize [spreadsort [upsampfac [errfail]]]]]]]",
    "\teg:\tfinufft2dmany_test 100 1e2 1e2 1e5 1e-6 1 0 0 2 0.0 1e-5",
    "\tnotes:\tif errfail present, exit code 1 if any error > errfail",
    NULL};
// Melody Shih Jun 2018; Barnett removed many_seq 7/27/18. Extra args 5/21/20.

int main(int argc, char *argv[]) {
  BIGINT M, N1, N2;           // M = # srcs, N1,N2 = # modes
  int ntransf;                // # of vectors for "many" interface
  double w, tol       = 1e-6; // default
  double err, errfail = INFINITY, errmax = 0;
  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  // opts.fftw = FFTW_MEASURE;  // change from default FFTW_ESTIMATE
  int isign = +1; // choose which exponential sign to test
  if (argc < 5 || argc > 12) {
    for (int i = 0; help[i]; ++i) fprintf(stderr, "%s\n", help[i]);
    return 2;
  }
  sscanf(argv[1], "%lf", &w);
  ntransf = (int)w;
  sscanf(argv[2], "%lf", &w);
  N1 = (BIGINT)w;
  sscanf(argv[3], "%lf", &w);
  N2 = (BIGINT)w;
  sscanf(argv[4], "%lf", &w);
  M = (BIGINT)w;
  if (argc > 5) sscanf(argv[5], "%lf", &tol);
  if (argc > 6) sscanf(argv[6], "%d", &opts.debug);
  opts.spread_debug = (opts.debug > 1) ? 1 : 0; // see output from spreader
  if (argc > 7) sscanf(argv[7], "%d", &opts.spread_thread);
  if (argc > 8) sscanf(argv[8], "%d", &opts.maxbatchsize);
  if (argc > 9) sscanf(argv[9], "%d", &opts.spread_sort);
  if (argc > 10) {
    sscanf(argv[10], "%lf", &w);
    opts.upsampfac = (FLT)w;
  }
  if (argc > 11) sscanf(argv[11], "%lf", &errfail);

  cout << scientific << setprecision(15);
  BIGINT N = N1 * N2;

  FLT *x = (FLT *)malloc(sizeof(FLT) * M);           // NU pts x coords
  FLT *y = (FLT *)malloc(sizeof(FLT) * M);           // NU pts y coords
  CPX *c = (CPX *)malloc(sizeof(CPX) * M * ntransf); // strengths
  CPX *F = (CPX *)malloc(sizeof(CPX) * N * ntransf); // mode ampls

#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT j = 0; j < M; ++j) {
      x[j] = M_PI * randm11r(&se);
      y[j] = M_PI * randm11r(&se);
    }
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT j = 0; j < ntransf * M; ++j) {
      c[j] = crandm11r(&se);
    }
  }

  printf("test 2d1 many vs repeated single: ------------------------------------\n");
  CNTime timer;
  timer.start();
  int ier   = FINUFFT2D1MANY(ntransf, M, x, y, c, isign, tol, N1, N2, F, &opts);
  double ti = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("ntr=%d: %lld NU pts to (%lld,%lld) modes in %.3g s \t%.3g NU pts/s\n",
           ntransf, (long long)M, (long long)N1, (long long)N2, ti, ntransf * M / ti);

  int i      = ntransf - 1; // choose a vector (transform number) to check
  BIGINT nt1 = (BIGINT)(0.37 * N1), nt2 = (BIGINT)(0.26 * N2); // choose some mode index
                                                               // to check
  CPX Ft = CPX(0, 0), J = IMA * (FLT)isign;
  for (BIGINT j = 0; j < M; ++j)
    Ft += c[j + i * M] * exp(J * (nt1 * x[j] + nt2 * y[j])); // crude direct
  BIGINT it = N1 / 2 + nt1 + N1 * (N2 / 2 + nt2); // index in complex F as 1d array
  err       = abs(Ft - F[it + i * N]) / infnorm(N, F + i * N);
  errmax    = max(err, errmax);
  printf("\tone mode: rel err in F[%lld,%lld] of trans#%d is %.3g\n", (long long)nt1,
         (long long)nt2, i, err);

  // compare the result with FINUFFT2D1
  FFTW_FORGET_WISDOM();
  finufft_opts simpleopts = opts;
  simpleopts.debug        = 0; // don't output timing for calls of FINUFFT2D1
  simpleopts.spread_debug = 0;

  CPX *cstart;
  CPX *Fstart;
  CPX *F_2d1 = (CPX *)malloc(sizeof(CPX) * N * ntransf);
  timer.restart();
  for (int k = 0; k < ntransf; ++k) {
    cstart = c + k * M;
    Fstart = F_2d1 + k * N;
    ier    = FINUFFT2D1(M, x, y, cstart, isign, tol, N1, N2, Fstart, &simpleopts);
  }
  double t = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("%d of: %lld NU pts to (%lld,%lld) modes in %.3g s \t%.3g NU pts/s\n", ntransf,
           (long long)M, (long long)N1, (long long)N2, t, ntransf * M / t);
  printf("\t\t\tspeedup \t T_FINUFFT2D1 / T_finufft2d1many = %.3g\n", t / ti);

  // Check consistency (worst over the ntransf)
  double maxerror = 0.0;
  for (int k = 0; k < ntransf; ++k)
    maxerror = max(maxerror, (double)relerrtwonorm(N, F_2d1 + k * N, F + k * N));
  errmax = max(maxerror, errmax);
  printf("\tconsistency check: sup ( ||f_many-f||_2 / ||f||_2  ) =  %.3g\n", maxerror);
  free(F_2d1);

  printf("test 2d2 many vs repeated single: ------------------------------------\n");

#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT m = 0; m < N * ntransf; ++m) F[m] = crandm11r(&se);
  }

  FFTW_FORGET_WISDOM();
  timer.restart();
  ier = FINUFFT2D2MANY(ntransf, M, x, y, c, isign, tol, N1, N2, F, &opts);
  ti  = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("ntr=%d: (%lld,%lld) modes to %lld NU pts in %.3g s \t%.3g NU pts/s\n",
           ntransf, (long long)N1, (long long)N2, (long long)M, ti, ntransf * M / ti);

  FFTW_FORGET_WISDOM();
  i         = ntransf - 1; // choose a data to check
  BIGINT jt = M / 2;       // check arbitrary choice of one targ pt
  CPX ct    = CPX(0, 0);
  BIGINT m  = 0;
  for (BIGINT m2 = -(N2 / 2); m2 <= (N2 - 1) / 2; ++m2) // loop in correct order over F
    for (BIGINT m1 = -(N1 / 2); m1 <= (N1 - 1) / 2; ++m1)
      ct += F[i * N + m++] * exp(J * (m1 * x[jt] + m2 * y[jt])); // crude direct
  err    = abs(ct - c[jt + i * M]) / infnorm(M, c + i * M);
  errmax = max(err, errmax);
  printf("\tone targ: rel err in c[%lld] of trans#%d is %.3g\n", (long long)jt, i, err);

  // compare the result with single calls to FINUFFT2D2...
  CPX *c_2d2 = (CPX *)malloc(sizeof(CPX) * M * ntransf);
  timer.restart();
  for (int k = 0; k < ntransf; ++k) {
    cstart = c_2d2 + k * M;
    Fstart = F + k * N;
    ier    = FINUFFT2D2(M, x, y, cstart, isign, tol, N1, N2, Fstart, &simpleopts);
  }
  t = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("%d of: (%lld,%lld) modes to %lld NU pts in %.3g s \t%.3g NU pts/s\n", ntransf,
           (long long)N1, (long long)N2, (long long)M, t, ntransf * M / t);
  printf("\t\t\tspeedup \t T_FINUFFT2D2 / T_finufft2d2many = %.3g\n", t / ti);

  maxerror = 0.0; // worst error over the ntransf
  for (int k = 0; k < ntransf; ++k)
    maxerror = max(maxerror, (double)relerrtwonorm(M, c_2d2 + k * M, c + k * M));
  errmax = max(maxerror, errmax);
  printf("\tconsistency check: sup ( ||c_many-c||_2 / ||c||_2 ) =  %.3g\n", maxerror);
  free(c_2d2);

  printf("test 2d3 many vs repeated single: ------------------------------------\n");
  FFTW_FORGET_WISDOM();

  // reuse the strengths c, interpret N as number of targs:
#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT j = 0; j < M; ++j) {
      x[j] = 2.0 + M_PI * randm11r(&se);  // new x_j srcs, offset from origin
      y[j] = -3.0 + M_PI * randm11r(&se); // " y_j
    }
  }

  FLT *s_freq = (FLT *)malloc(sizeof(FLT) * N); // targ freqs (1-cmpt)
  FLT *t_freq = (FLT *)malloc(sizeof(FLT) * N); // targ freqs (2-cmpt)
  FLT S1      = (FLT)N1 / 2;                    // choose freq range sim to type 1
  FLT S2      = (FLT)N2 / 2;

#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT k = 0; k < N; ++k) {
      s_freq[k] = S1 * (1.7 + randm11r(&se)); // S*(1.7 + k/(FLT)N); // offset the
                                              // freqs
      t_freq[k] = S2 * (-0.5 + randm11r(&se));
    }
  }

  timer.restart();
  ier = FINUFFT2D3MANY(ntransf, M, x, y, c, isign, tol, N, s_freq, t_freq, F, &opts);
  ti  = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("ntr=%d: %lld NU to %lld NU in %.3g s      \t%.3g tot NU pts/s\n", ntransf,
           (long long)M, (long long)N, ti, ntransf * (M + N) / ti);

  i         = ntransf - 1; // choose a transform to check
  BIGINT kt = N / 4;       // check arbitrary choice of one targ pt
  Ft        = CPX(0, 0);
  for (BIGINT j = 0; j < M; ++j)
    Ft += c[i * M + j] * exp(J * (s_freq[kt] * x[j] + t_freq[kt] * y[j]));
  err    = abs(Ft - F[kt + i * N]) / infnorm(N, F + i * N);
  errmax = max(err, errmax);
  printf("\tone targ: rel err in F[%lld] of trans#%d is %.3g\n", (long long)kt, i, err);

  // compare the result with FINUFFT2D3...
  FFTW_FORGET_WISDOM();
  CPX *f_2d3 = (CPX *)malloc(sizeof(CPX) * N * ntransf);
  timer.restart();
  for (int k = 0; k < ntransf; ++k) {
    Fstart = f_2d3 + k * N;
    cstart = c + k * M;
    ier = FINUFFT2D3(M, x, y, cstart, isign, tol, N, s_freq, t_freq, Fstart, &simpleopts);
  }
  t = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("%d of: %lld NU to %lld NU in %.3g s       \t%.3g tot NU pts/s\n", ntransf,
           (long long)M, (long long)N, t, ntransf * (M + N) / t);
  printf("\t\t\tspeedup \t T_FINUFFT2D3 / T_finufft2d3many = %.3g\n", t / ti);

  // check against the old
  maxerror = 0.0; // worst error over the ntransf
  for (int k = 0; k < ntransf; ++k)
    maxerror = max(maxerror, (double)relerrtwonorm(N, f_2d3 + k * N, F + k * N));
  errmax = max(maxerror, errmax);
  printf("\tconsistency check: sup ( ||f_many-f||_2 / ||f||_2 ) =  %.3g\n", maxerror);
  free(f_2d3);

  free(x);
  free(y);
  free(c);
  free(F);
  free(s_freq);
  free(t_freq);
  return (errmax > errfail);
}
