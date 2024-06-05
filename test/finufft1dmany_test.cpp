#include <finufft/test_defs.h>
// this enforces recompilation, responding to SINGLE...
#include "directft/dirft1d.cpp"
using namespace std;
using namespace finufft::utils;

const char *help[] = {
    "Tester for FINUFFT in 1d, vectorized, all 3 types, either precision.",
    "",
    "Usage: finufft1dmany_test ntrans Nmodes Nsrc [tol [debug [spread_thread "
    "[maxbatchsize [spreadsort [upsampfac [errfail]]]]]]]",
    "\teg:\tfinufft1dmany_test 100 1e3 1e4 1e-6 1 0 0 2 0.0 1e-5",
    "\tnotes:\tif errfail present, exit code 1 if any error > errfail",
    NULL};
// Malleo 2019 based on Shih 2018. Tidied, extra args, Barnett 5/25/20 onwards

int main(int argc, char *argv[]) {
  BIGINT M, N;                // M = # srcs, N = # modes
  int ntransf;                // # of vectors for "many" interface
  double w, tol       = 1e-6; // default
  double err, errfail = INFINITY, errmax = 0;
  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  // opts.fftw = FFTW_MEASURE;  // change from usual FFTW_ESTIMATE
  int isign = +1; // choose which exponential sign to test
  if (argc < 4 || argc > 11) {
    for (int i = 0; help[i]; ++i) fprintf(stderr, "%s\n", help[i]);
    return 2;
  }
  sscanf(argv[1], "%lf", &w);
  ntransf = (int)w;
  sscanf(argv[2], "%lf", &w);
  N = (BIGINT)w;
  sscanf(argv[3], "%lf", &w);
  M = (BIGINT)w;
  if (argc > 4) sscanf(argv[4], "%lf", &tol);
  if (argc > 5) sscanf(argv[5], "%d", &opts.debug);
  opts.spread_debug = (opts.debug > 1) ? 1 : 0; // see output from spreader
  if (argc > 6) sscanf(argv[6], "%d", &opts.spread_thread);
  if (argc > 7) sscanf(argv[7], "%d", &opts.maxbatchsize);
  if (argc > 8) sscanf(argv[8], "%d", &opts.spread_sort);
  if (argc > 9) {
    sscanf(argv[9], "%lf", &w);
    opts.upsampfac = (FLT)w;
  }
  if (argc > 10) sscanf(argv[10], "%lf", &errfail);

  cout << scientific << setprecision(15);

  FLT *x = (FLT *)malloc(sizeof(FLT) * M);           // NU pts x coords
  CPX *c = (CPX *)malloc(sizeof(CPX) * M * ntransf); // strengths
  CPX *F = (CPX *)malloc(sizeof(CPX) * N * ntransf); // mode ampls

#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT j = 0; j < M; ++j) {
      x[j] = M_PI * randm11r(&se);
    }
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT j = 0; j < ntransf * M; ++j) {
      c[j] = crandm11r(&se);
    }
  }

  printf("test 1d1 many vs repeated single: ------------------------------------\n");
  CNTime timer;
  timer.start();
  int ier   = FINUFFT1D1MANY(ntransf, M, x, c, isign, tol, N, F, &opts);
  double ti = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("ntr=%d: %lld NU pts to %lld modes in %.3g s  \t%.3g NU pts/s\n", ntransf,
           (long long)M, (long long)N, ti, ntransf * M / ti);

  int i      = (ntransf - 1);      // choose a trial to check
  BIGINT nt1 = (BIGINT)(0.37 * N); // choose some mode index to check
  CPX Ft = CPX(0, 0), J = IMA * (FLT)isign;
  for (BIGINT j = 0; j < M; ++j)
    Ft += c[j + i * M] * exp(J * (nt1 * x[j])); // crude direct
  BIGINT it = N / 2 + nt1;                      // index in complex F as 1d array
  err       = abs(Ft - F[it + i * N]) / infnorm(N, F + i * N);
  errmax    = max(err, errmax);
  printf("\tone mode: rel err in F[%lld] of trans#%d is %.3g\n", (long long)nt1, i, err);

  // compare the result with FINUFFT1D1
  FFTW_FORGET_WISDOM();
  CPX *F_1d1 = (CPX *)malloc(sizeof(CPX) * N * ntransf);
  CPX *Fstart;
  CPX *cstart;
  timer.restart();
  finufft_opts simpleopts = opts; // opts just for simple interface
  simpleopts.debug        = 0;
  simpleopts.spread_debug = 0;
  for (BIGINT j = 0; j < ntransf; j++) {
    Fstart = F_1d1 + j * N;
    cstart = c + j * M;
    FINUFFT1D1(M, x, cstart, isign, tol, N, Fstart, &simpleopts);
  }
  double t = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("%d of: %lld NU pts to %lld modes in %.3g s \t%.3g NU pts/s\n", ntransf,
           (long long)M, (long long)N, t, ntransf * M / t);
  printf("\t\t\tspeedup \t T_FINUFFT1D1 / T_finufft1d1many = %.3g\n", t / ti);

  // Check consistency (worst over the ntransf)
  double maxerror = 0.0;
  for (int k = 0; k < ntransf; ++k)
    maxerror = max(maxerror, (double)relerrtwonorm(N, F_1d1 + k * N, F + k * N));
  errmax = max(maxerror, errmax);
  printf("\tconsistency check: sup ( ||f_many-f||_2 / ||f||_2  ) =  %.3g\n", maxerror);
  free(F_1d1);

  printf("test 1d2 many vs repeated single: ------------------------------------\n");
  FFTW_FORGET_WISDOM();

#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM(); // needed for parallel random #s
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT m = 0; m < N; ++m) F[m] = crandm11r(&se);
  }
  timer.restart();
  ier = FINUFFT1D2MANY(ntransf, M, x, c, isign, tol, N, F, &opts);
  // cout<<"c:\n"; for (int j=0;j<M;++j) cout<<c[j]<<endl;
  ti = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("ntr=%d: %lld modes to %lld NU pts in %.3g s \t%.3g NU pts/s\n", ntransf,
           (long long)N, (long long)M, ti, ntransf * M / ti);

  BIGINT jt = M / 2;        // check arbitrary choice of one targ pt
  CPX ct    = CPX(0, 0);
  BIGINT m = 0, k0 = N / 2; // index shift in fk's = mag of most neg freq
  // #pragma omp parallel for schedule(static,TEST_RANDCHUNK) reduction(cmplxadd:ct)
  for (BIGINT m1 = -k0; m1 <= (N - 1) / 2; ++m1)
    ct += F[i * N + m++] * exp(IMA * ((FLT)(isign * m1)) * x[jt]); // crude direct
  err    = abs(ct - c[jt + i * M]) / infnorm(M, c + i * M);
  errmax = max(err, errmax);
  printf("\tone targ: rel err in c[%lld] of trans#%d is %.3g\n", (long long)jt, i, err);

  // check against single calls to FINUFFT1D2...
  FFTW_FORGET_WISDOM();
  CPX *c_1d2 = (CPX *)malloc(sizeof(CPX) * M * ntransf);
  timer.restart();
  for (BIGINT j = 0; j < ntransf; j++) {
    Fstart = F + j * N;
    cstart = c_1d2 + j * M;
    FINUFFT1D2(M, x, cstart, isign, tol, N, Fstart, &simpleopts);
  }
  t = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("%d of: %lld modes to %lld NU pts in %.3g s \t%.3g NU pts/s\n", ntransf,
           (long long)N, (long long)M, t, ntransf * M / t);
  printf("\t\t\tspeedup \t T_FINUFFT1D2 / T_finufft1d2many = %.3g\n", t / ti);

  maxerror = 0.0; // worst error over the ntransf
  for (int k = 0; k < ntransf; ++k)
    maxerror = max(maxerror, (double)relerrtwonorm(M, c_1d2 + k * M, c + k * M));
  errmax = max(maxerror, errmax);
  printf("\tconsistency check: sup ( ||c_many-c||_2 / ||c||_2 ) =  %.3g\n", maxerror);
  free(c_1d2);

  printf("test 1d3 many vs repeated single: ------------------------------------\n");
  FFTW_FORGET_WISDOM();

#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT j = 0; j < M; ++j) x[j] = 2.0 + PI * randm11r(&se); // new x_j srcs
  }
  FLT *s = (FLT *)malloc(sizeof(FLT) * N);                          // targ freqs
  FLT S  = (FLT)N / 2; // choose freq range sim to type 1
#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT k = 0; k < N; ++k)
      s[k] = S * (1.7 + randm11r(&se)); // S*(1.7 + k/(FLT)N); // offset

#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT j = 0; j < ntransf * M; ++j) c[j] = crandm11r(&se);
  }

  timer.restart();
  ier = FINUFFT1D3MANY(ntransf, M, x, c, isign, tol, N, s, F, &opts);
  ti  = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("ntr=%d: %lld NU to %lld NU in %.3g s       \t%.3g tot NU pts/s\n", ntransf,
           (long long)M, (long long)N, ti, ntransf * (M + N) / ti);

  BIGINT kt = N / 4; // check arbitrary choice of one targ pt
  Ft        = CPX(0, 0);
  // #pragma omp parallel for schedule(static,TEST_RANDCHUNK) reduction(cmplxadd:Ft)
  for (BIGINT j = 0; j < M; ++j)
    Ft += c[j + i * M] * exp(IMA * (FLT)isign * s[kt] * x[j]);
  err    = abs(Ft - F[kt + i * N]) / infnorm(N, F + i * N);
  errmax = max(err, errmax);
  printf("\tone targ: rel err in F[%lld] of trans#%d is %.3g\n", (long long)kt, i, err);

  // compare the result with single calls to FINUFFT1D3...
  FFTW_FORGET_WISDOM();
  CPX *f_1d3 = (CPX *)malloc(sizeof(CPX) * N * ntransf);
  timer.restart();
  for (int k = 0; k < ntransf; k++) {
    cstart = c + k * M;
    Fstart = f_1d3 + k * N;
    ier    = FINUFFT1D3(M, x, cstart, isign, tol, N, s, Fstart, &simpleopts);
  }
  t = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("%d of: %lld NU to %lld NU in %.3g s       \t%.3g tot NU pts/s\n", ntransf,
           (long long)M, (long long)N, t, ntransf * (M + N) / t);
  printf("\t\t\tspeedup \t T_FINUFFT1D3 / T_finufft1d3many = %.3g\n", t / ti);

  maxerror = 0.0; // worst error over the ntransf
  for (int k = 0; k < ntransf; ++k)
    maxerror = max(maxerror, (double)relerrtwonorm(N, f_1d3 + k * N, F + k * N));
  errmax = max(maxerror, errmax);
  printf("\tconsistency check: sup ( ||f_many-f||_2 / ||f||_2 ) =  %.3g\n", maxerror);
  free(f_1d3);
  free(x);
  free(s);
  free(c);
  free(F);
  return (errmax > errfail);
}
