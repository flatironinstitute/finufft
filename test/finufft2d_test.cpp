#include <finufft/test_defs.h>
// this enforces recompilation, responding to SINGLE...
#include "directft/dirft2d.cpp"
using namespace std;
using namespace finufft::utils;

const char *help[] = {"Tester for FINUFFT in 2d, all 3 types, either precision.",
                      "",
                      "Usage: finufft2d_test Nmodes1 Nmodes2 Nsrc [tol [debug "
                      "[spread_sort [upsampfac [errfail]]]]]",
                      "\teg:\tfinufft2d_test 1000 1000 1000000 1e-12 1 2 2.0 1e-11",
                      "\tnotes:\tif errfail present, exit code 1 if any error > errfail",
                      NULL};
// Barnett 2/1/17 onwards

int main(int argc, char *argv[]) {
  BIGINT M, N1, N2;           // M = # srcs, N1,N2 = # modes
  double w, tol       = 1e-6; // default
  double err, errfail = INFINITY, errmax = 0;
  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  // opts.fftw = FFTW_MEASURE;  // change from usual FFTW_ESTIMATE
  int isign = +1; // choose which exponential sign to test
  if (argc < 4 || argc > 9) {
    for (int i = 0; help[i]; ++i) fprintf(stderr, "%s\n", help[i]);
    return 2;
  }
  sscanf(argv[1], "%lf", &w);
  N1 = (BIGINT)w;
  sscanf(argv[2], "%lf", &w);
  N2 = (BIGINT)w;
  sscanf(argv[3], "%lf", &w);
  M = (BIGINT)w;
  if (argc > 4) sscanf(argv[4], "%lf", &tol);
  if (argc > 5) sscanf(argv[5], "%d", &opts.debug);
  opts.spread_debug = (opts.debug > 1) ? 1 : 0; // see output from spreader
  if (argc > 6) sscanf(argv[6], "%d", &opts.spread_sort);
  if (argc > 7) {
    sscanf(argv[7], "%lf", &w);
    opts.upsampfac = (FLT)w;
  }
  if (argc > 8) sscanf(argv[8], "%lf", &errfail);

  cout << scientific << setprecision(15);
  BIGINT N = N1 * N2;

  FLT *x = (FLT *)malloc(sizeof(FLT) * M); // NU pts x coords
  FLT *y = (FLT *)malloc(sizeof(FLT) * M); // NU pts y coords
  CPX *c = (CPX *)malloc(sizeof(CPX) * M); // strengths
  CPX *F = (CPX *)malloc(sizeof(CPX) * N); // mode ampls
#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM(); // needed for parallel random #s
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT j = 0; j < M; ++j) {
      x[j] = M_PI * randm11r(&se);
      y[j] = M_PI * randm11r(&se);
      c[j] = crandm11r(&se);
    }
  }

  printf("test 2d type 1:\n"); // -------------- type 1
  CNTime timer;
  timer.start();
  int ier   = FINUFFT2D1(M, x, y, c, isign, tol, N1, N2, F, &opts);
  double ti = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("\t%lld NU pts to (%lld,%lld) modes in %.3g s \t%.3g NU pts/s\n", (long long)M,
           (long long)N1, (long long)N2, ti, M / ti);

  BIGINT nt1 = (BIGINT)(0.37 * N1), nt2 = (BIGINT)(0.26 * N2); // choose some mode index
                                                               // to check
  FLT Ftr = 0, Fti = 0;                                        // crude direct...
#pragma omp parallel for schedule(static, TEST_RANDCHUNK) reduction(+ : Ftr, Fti)
  for (BIGINT j = 0; j < M; ++j) {            // Ft += c[j] * exp(J*(nt1*x[j]+nt2*y[j]))
    FLT z = (FLT)isign * (nt1 * x[j] + nt2 * y[j]), co = cos(z), si = sin(z);
    Ftr += real(c[j]) * co - imag(c[j]) * si; // cpx arith by hand
    Fti += imag(c[j]) * co + real(c[j]) * si;
  }
  BIGINT it = N1 / 2 + nt1 + N1 * (N2 / 2 + nt2); // index in complex F as 1d array
  err       = abs(Ftr + IMA * Fti - F[it]) / infnorm(N, F);
  printf("\tone mode: rel err in F[%lld,%lld] is %.3g\n", (long long)nt1, (long long)nt2,
         err);
  if ((int64_t)M * N <= TEST_BIGPROB) { // also check vs full direct eval
    CPX *Ft = (CPX *)malloc(sizeof(CPX) * N);
    dirft2d1(M, x, y, c, isign, N1, N2, Ft);
    err    = relerrtwonorm(N, Ft, F);
    errmax = max(err, errmax);
    printf("\tdirft2d: rel l2-err of result F is %.3g\n", err);
    free(Ft);
  } else
    errmax = max(err, errmax);

  printf("test 2d type 2:\n"); // -------------- type 2
#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT m = 0; m < N; ++m) F[m] = crandm11r(&se);
  }
  timer.restart();
  ier = FINUFFT2D2(M, x, y, c, isign, tol, N1, N2, F, &opts);
  ti  = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("\t(%lld,%lld) modes to %lld NU pts in %.3g s \t%.3g NU pts/s\n",
           (long long)N1, (long long)N2, (long long)M, ti, M / ti);

  BIGINT jt = M / 2; // check arbitrary choice of one targ pt
  CPX ct    = CPX(0, 0);
  BIGINT m  = 0;
  for (BIGINT m2 = -(N2 / 2); m2 <= (N2 - 1) / 2; ++m2) // loop in correct order over F
    for (BIGINT m1 = -(N1 / 2); m1 <= (N1 - 1) / 2; ++m1)
      ct += F[m++] * exp(IMA * (FLT)isign * (m1 * x[jt] + m2 * y[jt])); // crude
                                                                        // direct
  err = abs(ct - c[jt]) / infnorm(M, c);
  printf("\tone targ: rel err in c[%lld] is %.3g\n", (long long)jt, err);
  if ((int64_t)M * N <= TEST_BIGPROB) { // also full direct eval
    CPX *ct = (CPX *)malloc(sizeof(CPX) * M);
    dirft2d2(M, x, y, ct, isign, N1, N2, F);
    err    = relerrtwonorm(M, ct, c);
    errmax = max(err, errmax);
    printf("\tdirft2d: rel l2-err of result c is %.3g\n", err);
    // cout<<"c,ct:\n"; for (int j=0;j<M;++j) cout<<c[j]<<"\t"<<ct[j]<<endl;
    free(ct);
  } else
    errmax = max(err, errmax);

  printf("test 2d type 3:\n"); // -------------- type 3
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
  FLT *s = (FLT *)malloc(sizeof(FLT) * N); // targ freqs (1-cmpt)
  FLT *t = (FLT *)malloc(sizeof(FLT) * N); // targ freqs (2-cmpt)
  FLT S1 = (FLT)N1 / 2;                    // choose freq range sim to type 1
  FLT S2 = (FLT)N2 / 2;
#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT k = 0; k < N; ++k) {
      s[k] = S1 * (1.7 + randm11r(&se)); // S*(1.7 + k/(FLT)N); // offset the freqs
      t[k] = S2 * (-0.5 + randm11r(&se));
    }
  }
  timer.restart();
  ier = FINUFFT2D3(M, x, y, c, isign, tol, N, s, t, F, &opts);
  ti  = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("\t%lld NU to %lld NU in %.3g s         \t%.3g tot NU pts/s\n", (long long)M,
           (long long)N, ti, (M + N) / ti);

  BIGINT kt = N / 2;                          // check arbitrary choice of one targ pt
  Ftr = 0, Fti = 0;                           // crude direct...
#pragma omp parallel for schedule(static, TEST_RANDCHUNK) reduction(+ : Ftr, Fti)
  for (BIGINT j = 0; j < M; ++j) {            // Ft += exp(IMA*(FLT)isign*(s[kt]*x[j] +
                                              // t[kt]*y[j]))
    FLT z = (FLT)isign * (s[kt] * x[j] + t[kt] * y[j]), co = cos(z), si = sin(z);
    Ftr += real(c[j]) * co - imag(c[j]) * si; // cpx arith by hand
    Fti += imag(c[j]) * co + real(c[j]) * si;
  }
  err = abs(Ftr + IMA * Fti - F[kt]) / infnorm(N, F);
  printf("\tone targ: rel err in F[%lld] is %.3g\n", (long long)kt, err);
  if (((int64_t)M) * N <= TEST_BIGPROB) {     // also full direct eval
    CPX *Ft = (CPX *)malloc(sizeof(CPX) * N);
    dirft2d3(M, x, y, c, isign, N, s, t, Ft); // writes to F
    err    = relerrtwonorm(N, Ft, F);
    errmax = max(err, errmax);
    printf("\tdirft2d: rel l2-err of result F is %.3g\n", err);
    // cout<<"s t, F, Ft, F/Ft:\n"; for (int k=0;k<N;++k) cout<<s[k]<<" "<<t[k]<<",
    // "<<F[k]<<",\t"<<Ft[k]<<",\t"<<F[k]/Ft[k]<<endl;
    free(Ft);
  } else
    errmax = max(err, errmax);

  free(x);
  free(y);
  free(c);
  free(F);
  free(s);
  free(t);
  return (errmax > errfail);
}
