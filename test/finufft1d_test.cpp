#include <finufft/test_defs.h>
// this enforces recompilation, responding to SINGLE...
#include "directft/dirft1d.cpp"
using namespace std;
using namespace finufft::utils;

const char *help[] = {
    "Tester for FINUFFT in 1d, all 3 types, either precision.",
    "",
    "Usage: finufft1d_test Nmodes Nsrc [tol [debug [spread_sort [upsampfac [errfail]]]]]",
    "\teg:\tfinufft1d_test 1e6 1e6 1e-6 1 2 2.0 1e-5",
    "\tnotes:\tif errfail present, exit code 1 if any error > errfail",
    NULL};
// Barnett 1/22/17 onwards

int main(int argc, char *argv[]) {
  BIGINT M, N;                // M = # srcs, N = # modes out
  double w, tol       = 1e-6; // default
  double err, errfail = INFINITY, errmax = 0;
  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts); // put defaults in opts
  // opts.fftw = FFTW_MEASURE;  // change from usual FFTW_ESTIMATE
  int isign = +1; // choose which exponential sign to test
  if (argc < 3 || argc > 8) {
    for (int i = 0; help[i]; ++i) fprintf(stderr, "%s\n", help[i]);
    return 2;
  }
  sscanf(argv[1], "%lf", &w);
  N = (BIGINT)w;
  sscanf(argv[2], "%lf", &w);
  M = (BIGINT)w;
  if (argc > 3) sscanf(argv[3], "%lf", &tol);
  if (argc > 4) sscanf(argv[4], "%d", &opts.debug);
  opts.spread_debug = (opts.debug > 1) ? 1 : 0; // see output from spreader
  if (argc > 5) sscanf(argv[5], "%d", &opts.spread_sort);
  if (argc > 6) {
    sscanf(argv[6], "%lf", &w);
    opts.upsampfac = (FLT)w;
  }
  if (argc > 7) sscanf(argv[7], "%lf", &errfail);

  cout << scientific << setprecision(15);

  FLT *x = (FLT *)malloc(sizeof(FLT) * M); // NU pts
  CPX *c = (CPX *)malloc(sizeof(CPX) * M); // strengths
  CPX *F = (CPX *)malloc(sizeof(CPX) * N); // mode ampls
#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM();   // needed for parallel random #s
#pragma omp for schedule(static, TEST_RANDCHUNK) // static => non-stochastic
    for (BIGINT j = 0; j < M; ++j) {
      x[j] = PI * randm11r(&se);                 // fills [-pi,pi)
      c[j] = crandm11r(&se);
    }
  }
  // for (BIGINT j=0; j<M; ++j) x[j] = 0.999 * PI*randm11();  // avoid ends
  // for (BIGINT j=0; j<M; ++j) x[j] = PI*(2*j/(FLT)M-1);  // test a grid

  printf("test 1d type 1:\n"); // -------------- type 1
  CNTime timer;
  timer.start();
  int ier = FINUFFT1D1(M, x, c, isign, tol, N, F, &opts);
  // for (int j=0;j<N;++j) cout<<F[j]<<endl;
  double t = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("\t%lld NU pts to %lld modes in %.3g s \t%.3g NU pts/s\n", (long long)M,
           (long long)N, t, M / t);

  BIGINT nt = (BIGINT)(0.37 * N); // check arb choice of mode near the top (N/2)
  // #pragma omp declare reduction (cmplxadd:CPX:omp_out=omp_out+omp_in)
  // initializer(omp_priv={0.0,0.0})  // only for openmp v 4.0! #pragma omp parallel for
  // schedule(static,TEST_RANDCHUNK) reduction(cmplxadd:Ft)
  FLT Ftr = 0.0, Fti = 0.0;
#pragma omp parallel for schedule(static, TEST_RANDCHUNK) reduction(+ : Ftr, Fti)
  for (BIGINT j = 0; j < M; ++j) { // Ft += c[j] * exp(IMA*((FLT)(isign*nt))*x[j])
    FLT co = cos(((FLT)(isign * nt)) * x[j]), si = sin(((FLT)(isign * nt)) * x[j]);
    Ftr += real(c[j]) * co - imag(c[j]) * si; // cpx arith by hand
    Fti += imag(c[j]) * co + real(c[j]) * si;
  }
  err = abs(Ftr + IMA * Fti - F[N / 2 + nt]) / infnorm(N, F);
  printf("\tone mode: rel err in F[%lld] is %.3g\n", (long long)nt, err);
  if (((int64_t)M) * N <= TEST_BIGPROB) { // also full direct eval
    CPX *Ft = (CPX *)malloc(sizeof(CPX) * N);
    dirft1d1(M, x, c, isign, N, Ft);
    err    = relerrtwonorm(N, Ft, F);
    errmax = max(err, errmax);
    printf("\tdirft1d: rel l2-err of result F is %.3g\n", err);
    free(Ft);
  } else
    errmax = max(err, errmax);

  printf("test 1d type 2:\n"); // -------------- type 2
#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM(); // needed for parallel random #s
#pragma omp for schedule(static, TEST_RANDCHUNK)
    for (BIGINT m = 0; m < N; ++m) F[m] = crandm11r(&se);
  }
  timer.restart();
  ier = FINUFFT1D2(M, x, c, isign, tol, N, F, &opts);
  // cout<<"c:\n"; for (int j=0;j<M;++j) cout<<c[j]<<endl;
  t = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("\t%lld modes to %lld NU pts in %.3g s \t%.3g NU pts/s\n", (long long)N,
           (long long)M, t, M / t);

  BIGINT jt = M / 2;        // check arbitrary choice of one targ pt
  CPX ct    = CPX(0, 0);
  BIGINT m = 0, k0 = N / 2; // index shift in fk's = mag of most neg freq
  // #pragma omp parallel for schedule(static,TEST_RANDCHUNK) reduction(cmplxadd:ct)
  for (BIGINT m1 = -k0; m1 <= (N - 1) / 2; ++m1)
    ct += F[m++] * exp(IMA * ((FLT)(isign * m1)) * x[jt]); // crude direct
  err = abs(ct - c[jt]) / infnorm(M, c);
  printf("\tone targ: rel err in c[%lld] is %.3g\n", (long long)jt, err);
  if (((int64_t)M) * N <= TEST_BIGPROB) { // also full direct eval
    CPX *ct = (CPX *)malloc(sizeof(CPX) * M);
    dirft1d2(M, x, ct, isign, N, F);
    err    = relerrtwonorm(M, ct, c);
    errmax = max(err, errmax);
    printf("\tdirft1d: rel l2-err of result c is %.3g\n", err);
    // cout<<"c/ct:\n"; for (int j=0;j<M;++j) cout<<c[j]/ct[j]<<endl;
    free(ct);
  } else
    errmax = max(err, errmax);

  printf("test 1d type 3:\n"); // -------------- type 3
                               // reuse the strengths c, interpret N as number of targs:
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
  }
  timer.restart();
  ier = FINUFFT1D3(M, x, c, isign, tol, N, s, F, &opts);
  t   = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("\t%lld NU to %lld NU in %.3g s        \t%.3g tot NU pts/s\n", (long long)M,
           (long long)N, t, (M + N) / t);

  BIGINT kt = N / 2; // check arbitrary choice of one targ pt
  Ftr       = 0.0;
  Fti       = 0.0;
#pragma omp parallel for schedule(static, TEST_RANDCHUNK) reduction(+ : Ftr, Fti)
  for (BIGINT j = 0; j < M; ++j) { // Ft += c[j] * exp(IMA*(FLT)isign*s[kt]*x[j])
    FLT co = cos((FLT)isign * s[kt] * x[j]), si = sin((FLT)isign * s[kt] * x[j]);
    Ftr += real(c[j]) * co - imag(c[j]) * si; // cpx arith by hand
    Fti += imag(c[j]) * co + real(c[j]) * si;
  }
  err = abs(Ftr + IMA * Fti - F[kt]) / infnorm(N, F);
  printf("\tone targ: rel err in F[%lld] is %.3g\n", (long long)kt, err);
  if (((int64_t)M) * N <= TEST_BIGPROB) { // also full direct eval
    CPX *Ft = (CPX *)malloc(sizeof(CPX) * N);
    dirft1d3(M, x, c, isign, N, s, Ft);   // writes to F
    err    = relerrtwonorm(N, Ft, F);
    errmax = max(err, errmax);
    printf("\tdirft1d: rel l2-err of result F is %.3g\n", err);
    // cout<<"s, F, Ft:\n"; for (int k=0;k<N;++k) cout<<s[k]<<"
    // "<<F[k]<<"\t"<<Ft[k]<<"\t"<<F[k]/Ft[k]<<endl;
    free(Ft);
  } else
    errmax = max(err, errmax);

  free(x);
  free(c);
  free(F);
  free(s);
  return (errmax > errfail);
}
