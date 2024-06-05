#include <finufft/test_defs.h>
// this enforces recompilation, responding to SINGLE...
#include "directft/dirft3d.cpp"
using namespace std;
using namespace finufft::utils;

const char *help[] = {
    "Tester for FINUFFT in 3d, vectorized, all 3 types, either precision.",
    "",
    "Usage: finufft3dmany_test ntrans Nmodes1 Nmodes2 Nmodes3 Nsrc [tol [debug "
    "[spread_thread [maxbatchsize [spreadsort [upsampfac [errfail]]]]]]]",
    "\teg:\tfinufft3dmany_test 100 50 50 50 1e5 1e-3 1 0 0 2 0.0 1e-2",
    "\tnotes:\tif errfail present, exit code 1 if any error > errfail",
    NULL};
// Malleo 2019 based on Shih 2018. Tidied, extra args, Barnett 5/25/20.

int main(int argc, char *argv[]) {
  BIGINT M, N1, N2, N3;       // M = # srcs, N1,N2 = # modes
  int ntransf;                // # of vectors for "many" interface
  double w, tol       = 1e-6; // default
  double err, errfail = INFINITY, errmax = 0;
  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  // opts.fftw = FFTW_MEASURE;  // change from usual FFTW_ESTIMATE
  int isign = +1; // choose which exponential sign to test
  if (argc < 6 || argc > 13) {
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
  N3 = (BIGINT)w;
  sscanf(argv[5], "%lf", &w);
  M = (BIGINT)w;
  if (argc > 6) sscanf(argv[6], "%lf", &tol);
  if (argc > 7) sscanf(argv[7], "%d", &opts.debug);
  opts.spread_debug = (opts.debug > 1) ? 1 : 0; // see output from spreader
  if (argc > 8) sscanf(argv[8], "%d", &opts.spread_thread);
  if (argc > 9) sscanf(argv[9], "%d", &opts.maxbatchsize);
  if (argc > 10) sscanf(argv[10], "%d", &opts.spread_sort);
  if (argc > 11) {
    sscanf(argv[11], "%lf", &w);
    opts.upsampfac = (FLT)w;
  }
  if (argc > 12) sscanf(argv[12], "%lf", &errfail);

  cout << scientific << setprecision(15);
  BIGINT N = N1 * N2 * N3;

  FLT *x = (FLT *)malloc(sizeof(FLT) * M);           // NU pts x coords
  FLT *y = (FLT *)malloc(sizeof(FLT) * M);           // NU pts y coords
  FLT *z = (FLT *)malloc(sizeof(FLT) * M);           // NU pts z coords
  CPX *c = (CPX *)malloc(sizeof(CPX) * M * ntransf); // strengths
  CPX *F = (CPX *)malloc(sizeof(CPX) * N * ntransf); // mode ampls

#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT j = 0; j < M; ++j) {
      x[j] = M_PI * randm11r(&se);
      y[j] = M_PI * randm11r(&se);
      z[j] = M_PI * randm11r(&se);
    }
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT j = 0; j < ntransf * M; ++j) {
      c[j] = crandm11r(&se);
    }
  }

  printf("test 3d1 many vs repeated single: ------------------------------------\n");
  CNTime timer;
  timer.start();
  int ier   = FINUFFT3D1MANY(ntransf, M, x, y, z, c, isign, tol, N1, N2, N3, F, &opts);
  double ti = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("ntr=%d: %lld NU pts to (%lld,%lld,%lld) modes in %.3g s \t%.3g NU pts/s\n",
           ntransf, (long long)M, (long long)N1, (long long)N2, (long long)N3, ti,
           ntransf * M / ti);

  int i      = ntransf - 1;          // choose a data to check
  BIGINT nt1 = (BIGINT)(0.37 * N1), nt2 = (BIGINT)(0.26 * N2),
         nt3 = (BIGINT)(-0.39 * N3); // choose some mode index to check
  CPX Ft = CPX(0, 0), J = IMA * (FLT)isign;
  for (BIGINT j = 0; j < M; ++j)
    Ft += c[j + i * M] * exp(J * (nt1 * x[j] + nt2 * y[j] + nt3 * z[j]));    // crude
                                                                             // direct
  BIGINT it = N1 / 2 + nt1 + N1 * (N2 / 2 + nt2) + N1 * N2 * (N3 / 2 + nt3); // index in
                                                                             // complex
                                                                             // F as 1d
                                                                             // array
  err    = abs(Ft - F[it + i * N]) / infnorm(N, F + i * N);
  errmax = max(err, errmax);
  printf("\tone mode: rel err in F[%lld,%lld,%lld] of trans#%d is %.3g\n", (long long)nt1,
         (long long)nt2, (long long)nt3, i, err);

  // compare the result with FINUFFT3D1
  FFTW_FORGET_WISDOM();
  finufft_opts simpleopts = opts;
  simpleopts.debug        = 0; // don't output timing for calls of FINUFFT3D1
  simpleopts.spread_debug = 0;

  CPX *cstart;
  CPX *Fstart;
  CPX *F_3d1 = (CPX *)malloc(sizeof(CPX) * N * ntransf);
  timer.restart();
  for (int k = 0; k < ntransf; ++k) {
    cstart = c + k * M;
    Fstart = F_3d1 + k * N;
    ier    = FINUFFT3D1(M, x, y, z, cstart, isign, tol, N1, N2, N3, Fstart, &simpleopts);
  }
  double t = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("%d of: %lld NU pts to (%lld,%lld,%lld) modes in %.3g s  \t%.3g NU pts/s\n",
           ntransf, (long long)M, (long long)N1, (long long)N2, (long long)N3, t,
           ntransf * M / t);
  printf("\t\t\tspeedup \t T_FINUFFT3D1 / T_finufft3d1many = %.3g\n", t / ti);

  // Check accuracy (worst over the ntransf)
  double maxerror = 0.0;
  for (int k = 0; k < ntransf; ++k)
    maxerror = max(maxerror, (double)relerrtwonorm(N, F_3d1 + k * N, F + k * N));
  errmax = max(maxerror, errmax);
  printf("\tconsistency check: sup ( ||f_many-f||_2 / ||f||_2 ) =  %.3g\n", maxerror);
  free(F_3d1);

  printf("test 3d2 many vs repeated single: ------------------------------------\n");
#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT m = 0; m < N * ntransf; ++m) F[m] = crandm11r(&se);
  }
  FFTW_FORGET_WISDOM();
  timer.restart();
  ier = FINUFFT3D2MANY(ntransf, M, x, y, z, c, isign, tol, N1, N2, N3, F, &opts);
  ti  = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("ntr=%d: (%lld,%lld,%lld) modes to %lld NU pts in %.3g s \t%.3g NU pts/s\n",
           ntransf, (long long)N1, (long long)N2, (long long)N3, (long long)M, ti,
           ntransf * M / ti);

  i         = ntransf - 1; // choose a data to check
  BIGINT jt = M / 2;       // check arbitrary choice of one targ pt
  CPX ct    = CPX(0, 0);
  BIGINT m  = 0;
  for (BIGINT m3 = -(N3 / 2); m3 <= (N3 - 1) / 2; ++m3) {
    for (BIGINT m2 = -(N2 / 2); m2 <= (N2 - 1) / 2; ++m2) { // loop in correct order
                                                            // over F
      for (BIGINT m1 = -(N1 / 2); m1 <= (N1 - 1) / 2; ++m1) {
        ct += F[i * N + m++] * exp(J * (m1 * x[jt] + m2 * y[jt] + m3 * z[jt])); // crude
                                                                                // direct
      }
    }
  }
  err    = abs(ct - c[jt + i * M]) / infnorm(M, c + i * M);
  errmax = max(err, errmax);
  printf("\tone targ: rel err in c[%lld] of trans#%d is %.3g\n", (long long)jt, i, err);

  FFTW_FORGET_WISDOM();
  // compare the result with FINUFFT3D2...
  CPX *c_3d2 = (CPX *)malloc(sizeof(CPX) * M * ntransf);
  timer.restart();
  for (int k = 0; k < ntransf; ++k) {
    cstart = c_3d2 + k * M;
    Fstart = F + k * N;
    ier    = FINUFFT3D2(M, x, y, z, cstart, isign, tol, N1, N2, N3, Fstart, &simpleopts);
  }
  t = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("%d of: (%lld,%lld,%lld) modes to %lld NU pts in %.3g s \t%.3g NU pts/s\n",
           ntransf, (long long)N1, (long long)N2, (long long)N3, (long long)M, t,
           ntransf * M / t);
  printf("\t\t\tspeedup \t T_FINUFFT3D2 / T_finufft3d2many = %.3g\n", t / ti);

  maxerror = 0.0; // worst error over the ntransf
  for (int k = 0; k < ntransf; ++k)
    maxerror = max(maxerror, (double)relerrtwonorm(M, c_3d2 + k * M, c + k * M));
  errmax = max(maxerror, errmax);
  printf("\tconsistency check: sup ( ||c_many-c||_2 / ||c||_2 ) =  %.3g\n", maxerror);
  free(c_3d2);

  printf("test 3d3 many vs repeated single: ------------------------------------\n");
  FFTW_FORGET_WISDOM();
  // reuse the strengths c, interpret N as number of targs:
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
  FLT *s_freq = (FLT *)malloc(sizeof(FLT) * N); // targ freqs (1-cmpt)
  FLT *t_freq = (FLT *)malloc(sizeof(FLT) * N); // targ freqs (2-cmpt)
  FLT *u_freq = (FLT *)malloc(sizeof(FLT) * N); // targ freqs (3-cmpt)
  FLT S1      = (FLT)N1 / 2;                    // choose freq range sim to type 1
  FLT S2      = (FLT)N2 / 2;
  FLT S3      = (FLT)N3 / 2;

#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT k = 0; k < N; ++k) {
      s_freq[k] = S1 * (1.7 + randm11r(&se)); // S*(1.7 + k/(FLT)N); // offset the
                                              // freqs
      t_freq[k] = S2 * (-0.5 + randm11r(&se));
      u_freq[k] = S3 * (0.9 + randm11r(&se));
    }
  }

  timer.restart();
  ier = FINUFFT3D3MANY(ntransf, M, x, y, z, c, isign, tol, N, s_freq, t_freq, u_freq, F,
                       &opts);
  ti  = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("ntr=%d: %lld NU to %lld NU in %.3g s \t%.3g tot NU pts/s\n", ntransf,
           (long long)M, (long long)N, ti, ntransf * (M + N) / ti);

  i         = ntransf - 1; // choose a transform to check
  BIGINT kt = N / 4;       // check arbitrary choice of one targ pt
  Ft        = CPX(0, 0);
  for (BIGINT j = 0; j < M; ++j)
    Ft += c[i * M + j] *
          exp(J * (s_freq[kt] * x[j] + t_freq[kt] * y[j] + u_freq[kt] * z[j]));
  err    = abs(Ft - F[kt + i * N]) / infnorm(N, F + i * N);
  errmax = max(err, errmax);
  printf("\t one targ: rel err in F[%lld] of trans#%d is %.3g\n", (long long)kt, i, err);

  FFTW_FORGET_WISDOM();
  // compare the result with FINUFFT3D3...
  CPX *f_3d3 = (CPX *)malloc(sizeof(CPX) * N * ntransf);
  timer.restart();
  for (int k = 0; k < ntransf; ++k) {
    Fstart = f_3d3 + k * N;
    cstart = c + k * M;
    ier    = FINUFFT3D3(M, x, y, z, cstart, isign, tol, N, s_freq, t_freq, u_freq, Fstart,
                        &simpleopts);
  }
  t = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("%d of: %lld NU to %lld NU in %.3g s   \t%.3g tot NU pts/s\n", ntransf,
           (long long)M, (long long)N, t, ntransf * (M + N) / t);
  printf("\t\t\tspeedup \t T_FINUFFT3D3 / T_finufft3d3many = %.3g\n", t / ti);

  maxerror = 0.0; // worst error over the ntransf
  for (int k = 0; k < ntransf; ++k)
    maxerror = max(maxerror, (double)relerrtwonorm(N, f_3d3 + k * N, F + k * N));
  errmax = max(maxerror, errmax);
  printf("\tconsistency check: sup ( ||f_many-f||_2 / ||f||_2 ) =  %.3g\n", maxerror);
  free(f_3d3);

  free(x);
  free(y);
  free(z);
  free(c);
  free(F);
  free(s_freq);
  free(t_freq);
  free(u_freq);
  return (errmax > errfail);
}
