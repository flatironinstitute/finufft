/* Tester calling FINUFFT library from C++ using all manner of crazy inputs and
   edge cases that might cause errors, and to check those errors give the right
   error codes gracefully, and the right answers if relevant.

   Simple and "many" interfaces mostly, with two guru cases at the end.

   Usage (linux):  ./dumbinputs{f} 2> /dev/null
   (since FINUFFT will spit msgs to stderr, to be ignored)

   Pass: exit code 0. (Stdout should indicate passed)
   Fail: exit code>0. (Stdout may indicate what failed)

   Notes: due to large number of FINUFFT calls, nthreads is set low internally.
   (usually much faster than a large number of threads)

   To do: add more tests, eg, of invalid opts params.

   Barnett 3/14/17, updated Andrea Malleo, summer 2019.
   Libin Lu switch to use ptr-to-opts interfaces, Feb 2020.
   guru: makeplan followed by immediate destroy. Barnett 5/26/20.
   Either precision with dual-prec lib funcs 7/3/20.
   Added a chkbnds case to 1d1, 4/9/21.
   Made pass-fail, obviating results/dumbinputs.refout. Barnett 6/16/23.
   Removed the chkbnds case to 1d1, 05/08/2024.

   Suggested compile:
   g++ -std=c++14 -fopenmp dumbinputs.cpp -I../include ../lib/libfinufft.so -o dumbinputs
   -lfftw3 -lfftw3_omp -lm g++ -std=c++14 -fopenmp dumbinputs.cpp -I../include
   ../lib/libfinufft.so -o dumbinputsf -lfftw3 -lfftw3_omp -lm -DSINGLE

   or if you have built a single-core version:
   g++ -std=c++14 dumbinputs.cpp -I../include ../lib/libfinufft.so -o dumbinputs -lfftw3
   -lm etc
*/

// This switches FLT macro from double to float if SINGLE is defined, etc...
#include "directft/dirft1d.cpp"
#include "directft/dirft2d.cpp"
#include "directft/dirft3d.cpp"
#include <finufft/test_defs.h>
using namespace std;
using namespace finufft::utils; // for twonorm, etc

int main(int argc, char *argv[]) {
  int M = 100;    // number of nonuniform points
  int N = 10;     // # modes, keep small, also output NU pts in type 3
#ifdef SINGLE
  FLT acc = 1e-5; // desired accuracy for NUFFTs  (prec-dep)
#else
  FLT acc = 1e-8; // desired accuracy for NUFFTs
#endif
  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);

  int NN = N * N * N; // modes F alloc size since we'll go to 3d
  // generate some "random" nonuniform points (x) and complex strengths (c):
  FLT *x = (FLT *)malloc(sizeof(FLT) * M);
  CPX *c = (CPX *)malloc(sizeof(CPX) * M);
  for (int j = 0; j < M; ++j) {
    x[j] = PI * cos((FLT)j); // deterministic
    c[j] = sin((FLT)1.3 * j) + IMA * cos((FLT)0.9 * j);
  }
  // allocate output array F for Fourier modes, fix some type-3 coords...
  CPX *F = (CPX *)malloc(sizeof(CPX) * NN);
  FLT *s = (FLT *)malloc(sizeof(FLT) * N);
  for (int k = 0; k < N; ++k) s[k] = 10 * cos(1.2 * k); // normal-sized coords
  FLT *shuge = (FLT *)malloc(sizeof(FLT) * N);
  FLT huge   = 1e12;                                    // no smaller than MAX_NF
  for (int k = 0; k < N; ++k) shuge[k] = huge * s[k];   // some huge coords

  // alloc exact output array
  CPX *Fe = (CPX *)malloc(sizeof(CPX) * NN);

  // some useful debug printing...
  // for (int k=0;k<N;++k) printf("F[%d] = %g+%gi\n",k,real(F[k]),imag(F[k]));
  // for (int j=0;j<M;++j) printf("c[%d] = %g+%gi\n",j,real(c[j]),imag(c[j]));
  // printf("%.3g %3g\n",twonorm(N,F),twonorm(M,c));
  opts.debug        = 0; // set to 1,2, to debug inside FINUFFT, etc segfaults
  opts.spread_debug = 0;

  opts.nthreads = 1; // to keep them fast (thread-launch is slow)

#ifdef SINGLE
  printf("dumbinputsf test start...\n");
#else
  printf("dumbinputs test start...\n");
#endif

  // 111111111111111111111111111111111111111111111111111111111111111111111111
  printf("1D dumb cases.\n");
  int ier = FINUFFT1D1(M, x, c, +1, 0, N, F, &opts);
  if (ier != FINUFFT_WARN_EPS_TOO_SMALL) {
    printf("1d1 tol=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT1D1(M, x, c, +1, acc, 0, F, &opts);
  if (ier) {
    printf("1d1 N=0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT1D1(-1, x, c, +1, acc, 0, F, &opts);
  if (ier != FINUFFT_ERR_NUM_NU_PTS_INVALID) {
    printf("1d1 M<0:\twrong err code %d\n", ier);
    return 1;
  }
  int64_t Mhuge = (int64_t)(1e16); // cf defs.h MAX_NU_PTS
  ier           = FINUFFT1D1(Mhuge, x, c, +1, acc, 0, F, &opts);
  if (ier != FINUFFT_ERR_NUM_NU_PTS_INVALID) {
    printf("1d1 M huge:\twrong err code %d\n", ier);
    return 1;
  }
  ier   = FINUFFT1D1(0, x, c, +1, acc, N, F, &opts);
  FLT t = twonorm(N, F);
  if (ier || t != 0.0) {
    printf("1d1 M=0:\tier=%d nrm(F)=%.3g\n", ier, t);
    return 1;
  }
  for (int k = 0; k < NN; ++k)
    F[k] = sin((FLT)0.7 * k) + IMA * cos((FLT)0.3 * k); // set F for t2
  ier = FINUFFT1D2(M, x, c, +1, 0, N, F, &opts);
  if (ier != FINUFFT_WARN_EPS_TOO_SMALL) {
    printf("1d2 tol=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT1D2(M, x, c, +1, acc, 0, F, &opts);
  t   = twonorm(M, c);
  if (ier || t != 0.0) {
    printf("1d2 N=0:\tier=%d nrm(c)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT1D2(0, x, c, +1, acc, N, F, &opts);
  if (ier) {
    printf("1d2 M=0:\tier=%d\n", ier);
    return ier;
  }
  for (int j = 0; j < M; ++j)
    c[j] = sin((FLT)1.3 * j) + IMA * cos((FLT)0.9 * j); // reset c for t3
  ier = FINUFFT1D3(M, x, c, +1, 0, N, s, F, &opts);
  if (ier != FINUFFT_WARN_EPS_TOO_SMALL) {
    printf("1d3 tol=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT1D3(M, x, c, +1, acc, 0, s, F, &opts);
  if (ier) {
    printf("1d3 nk=0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT1D3(M, x, c, +1, acc, -1, s, F, &opts);
  if (ier != FINUFFT_ERR_NUM_NU_PTS_INVALID) {
    printf("1d3 nk=-1:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT1D3(M, x, c, +1, acc, Mhuge, s, F, &opts);
  if (ier != FINUFFT_ERR_NUM_NU_PTS_INVALID) {
    printf("1d3 nk huge:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT1D3(0, x, c, +1, acc, N, s, F, &opts);
  t   = twonorm(N, F);
  if (ier || t != 0.0) {
    printf("1d3 M=0:\tier=%d nrm(F)=%.3g\n", ier, t);
    return 1;
  }
  // for type 3 only we include crude accuracy check for 1-NUpt (I/O) cases...
  ier = FINUFFT1D3(1, x, c, +1, acc, N, s, F, &opts); // XK prod formally 0
  dirft1d3(1, x, c, +1, N, s, Fe);
  for (int k = 0; k < N; ++k) F[k] -= Fe[k];          // acc chk
  FLT err = twonorm(N, F) / sqrt((FLT)N);
  if (ier || err > 100 * acc) {
    printf("1d3 M=1:\tier=%d nrm(err)=%.3g\n", ier, err);
    return 1;
  }
  ier = FINUFFT1D3(M, x, c, +1, acc, 1, s, F, &opts);
  dirft1d3(M, x, c, +1, 1, s, Fe);
  err = abs(F[0] - Fe[0]);
  if (ier || err > 10 * acc) {
    printf("1d3 nk=1:\tier=%d err=%.3g\n", ier, err);
    return 1;
  }
  ier = FINUFFT1D3(1, x, c, +1, acc, 1, s, F, &opts);
  dirft1d3(1, x, c, +1, 1, s, Fe);
  err = abs(F[0] - Fe[0]);
  if (ier || err > 10 * acc) {
    printf("1d3 M=nk=1:\tier=%d err=%.3g\n", ier, err);
    return 1;
  }
  ier = FINUFFT1D3(M, x, c, +1, acc, N, shuge, F, &opts);
  if (ier == 0) { // any nonzero code accepted here
    printf("1d3 XK prod too big:\twrong error code %d\n", ier);
    return 1;
  }
  int ndata = 10; // how many multiple vectors to test it on
  CPX *cm   = (CPX *)malloc(sizeof(CPX) * M * ndata);
  CPX *Fm   = (CPX *)malloc(sizeof(CPX) * NN * ndata);   // the biggest array
  for (int j = 0; j < M * ndata; ++j)
    cm[j] = sin((FLT)1.3 * j) + IMA * cos((FLT)0.9 * j); // set cm for 1d1many
  ier = FINUFFT1D1MANY(0, M, x, cm, +1, 0, N, Fm, &opts);
  if (ier != FINUFFT_ERR_NTRANS_NOTVALID) {
    printf("1d1many ndata=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT1D1MANY(ndata, M, x, cm, +1, 0, N, Fm, &opts);
  if (ier != FINUFFT_WARN_EPS_TOO_SMALL) {
    printf("1d1many tol=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT1D1MANY(ndata, M, x, cm, +1, acc, 0, Fm, &opts);
  if (ier) {
    printf("1d1many N=0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT1D1MANY(ndata, 0, x, cm, +1, acc, N, Fm, &opts);
  t   = twonorm(N * ndata, Fm);
  if (ier || t != 0.0) {
    printf("1d1many M=0:\tier=%d nrm(Fm)=%.3g\n", ier, t);
    return 1;
  }
  for (int k = 0; k < NN * ndata; ++k)
    Fm[k] = sin((FLT)0.7 * k) + IMA * cos((FLT)0.3 * k); // set Fm for 1d2many
  ier = FINUFFT1D2MANY(0, M, x, cm, +1, 0, N, Fm, &opts);
  if (ier != FINUFFT_ERR_NTRANS_NOTVALID) {
    printf("1d2many ndata=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT1D2MANY(ndata, M, x, cm, +1, 0, N, Fm, &opts);
  if (ier != FINUFFT_WARN_EPS_TOO_SMALL) {
    printf("1d2many tol=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT1D2MANY(ndata, M, x, cm, +1, acc, 0, Fm, &opts);
  t   = twonorm(N * ndata, cm);
  if (ier || t != 0.0) {
    printf("1d2many N=0:\tier=%d nrm(cm)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT1D2MANY(ndata, 0, x, cm, +1, acc, N, Fm, &opts);
  if (ier) {
    printf("1d2many M=0:\tier=%d\n", ier);
    return ier;
  }
  for (int j = 0; j < M * ndata; ++j)
    cm[j] = sin((FLT)1.3 * j) + IMA * cos((FLT)0.9 * j); // reset cm for 1d3many
  ier = FINUFFT1D3MANY(0, M, x, cm, +1, acc, N, s, Fm, &opts);
  if (ier != FINUFFT_ERR_NTRANS_NOTVALID) {
    printf("1d3many ndata=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT1D3MANY(ndata, M, x, cm, +1, 0, N, s, Fm, &opts);
  if (ier != FINUFFT_WARN_EPS_TOO_SMALL) {
    printf("1d3many tol=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT1D3MANY(ndata, M, x, cm, +1, acc, 0, s, Fm, &opts);
  if (ier) {
    printf("1d3many nk=0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT1D3MANY(ndata, 0, x, cm, +1, acc, N, s, Fm, &opts);
  t   = twonorm(N, Fm);
  // again, as above, only crude acc tests for 1-NUpt (I/O) case...
  ier = FINUFFT1D3MANY(ndata, 1, x, cm, +1, acc, N, s, Fm, &opts); // XK prod formally 0
  dirft1d3(1, x, c, +1, N, s, Fe);
  for (int k = 0; k < N; ++k) Fm[k] -= Fe[k];                      // acc chk
  err = twonorm(N, Fm) / sqrt((FLT)N); // rms, to 5e-5 abs; check just first trial
  if (ier || err > 100 * acc) {
    printf("1d3many M=1:\tier=%d nrm(err)=%.3g\n", ier, err);
    return 1;
  }
  ier = FINUFFT1D3MANY(ndata, M, x, cm, +1, acc, 1, s, Fm, &opts);
  dirft1d3(M, x, c, +1, 1, s, Fe);
  err = abs(Fm[0] - Fe[0]);
  if (ier || err > 10 * acc) {
    printf("1d3many nk=1:\tier=%d err=%.3g\n", ier, err);
    return 1;
  }
  ier = FINUFFT1D3MANY(ndata, 1, x, cm, +1, acc, 1, s, Fm, &opts);
  dirft1d3(1, x, c, +1, 1, s, Fe);
  err = abs(Fm[0] - Fe[0]);
  if (ier || err > 10 * acc) {
    printf("1d3many M=nk=1:\tier=%d err=%.3g\n", ier, err);
    return 1;
  }
  ier = FINUFFT1D3MANY(ndata, M, x, cm, +1, acc, N, shuge, Fm, &opts);
  if (ier == 0) { // any nonzero code accepted here
    printf("1d3many XK prod too big:\twrong error code %d\n", ier);
    return 1;
  }

  // 2222222222222222222222222222222222222222222222222222222222222222222222222
  printf("2D dumb cases.\n"); // (uses y=x, and t=s in type 3)
  ier = FINUFFT2D1(M, x, x, c, +1, 0, N, N, F, &opts);
  if (ier != FINUFFT_WARN_EPS_TOO_SMALL) {
    printf("2d1 tol=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT2D1(M, x, x, c, +1, acc, 0, 0, F, &opts);
  if (ier) {
    printf("2d1 Ns=Nt=0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT2D1(M, x, x, c, +1, acc, 0, N, F, &opts);
  if (ier) {
    printf("2d1 Ns=0,Nt>0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT2D1(M, x, x, c, +1, acc, N, 0, F, &opts);
  if (ier) {
    printf("2d1 Ns>0,Nt=0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT2D1(0, x, x, c, +1, acc, N, N, F, &opts);
  t   = twonorm(N, F);
  if (ier || t != 0.0) {
    printf("2d1 M=0:\tier=%d nrm(F)=%.3g\n", ier, t);
    return 1;
  }
  for (int k = 0; k < NN; ++k)
    F[k] = sin((FLT)0.7 * k) + IMA * cos((FLT)0.3 * k); // set F for t2
  ier = FINUFFT2D2(M, x, x, c, +1, 0, N, N, F, &opts);
  if (ier != FINUFFT_WARN_EPS_TOO_SMALL) {
    printf("2d2 tol=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT2D2(M, x, x, c, +1, acc, 0, 0, F, &opts);
  t   = twonorm(M, c);
  if (ier || t != 0.0) {
    printf("2d2 Ns=Nt=0:\tier=%d nrm(c)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT2D2(M, x, x, c, +1, acc, 0, N, F, &opts);
  t   = twonorm(M, c);
  if (ier || t != 0.0) {
    printf("2d2 Ns=0,Nt>0:\tier=%d nrm(c)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT2D2(M, x, x, c, +1, acc, N, 0, F, &opts);
  t   = twonorm(M, c);
  if (ier || t != 0.0) {
    printf("2d2 Ns>0,Nt=0:\tier=%d nrm(c)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT2D2(0, x, x, c, +1, acc, N, N, F, &opts);
  if (ier) {
    printf("2d2 M=0:\tier=%d\n", ier);
    return ier;
  }
  for (int j = 0; j < M; ++j)
    c[j] = sin((FLT)1.3 * j) + IMA * cos((FLT)0.9 * j); // reset c for t3
  ier = FINUFFT2D3(M, x, x, c, +1, 0, N, s, s, F, &opts);
  if (ier != FINUFFT_WARN_EPS_TOO_SMALL) {
    printf("2d3 tol=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT2D3(M, x, x, c, +1, acc, 0, s, s, F, &opts);
  if (ier) {
    printf("2d3 nk=0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT2D3(0, x, x, c, +1, acc, N, s, s, F, &opts);
  t   = twonorm(N, F);
  if (ier || t != 0.0) {
    printf("2d3 M=0:\tier=%d nrm(F)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT2D3(1, x, x, c, +1, acc, N, s, s, F, &opts); // XK prod formally 0
  // we don't check the M=nk=1 case for >1D since guess that 1D would catch it.
  if (ier) {
    printf("2d3 M=nk=1:\tier=%d\n", ier);
    return ier;
  }
  for (int k = 0; k < N; ++k) shuge[k] = sqrt(huge) * s[k]; // less huge coords
  ier = FINUFFT2D3(M, x, x, c, +1, acc, N, shuge, shuge, F, &opts);
  if (ier == 0) { // any nonzero code accepted here
    printf("2d3 XK prod too big:\twrong error code %d\n", ier);
    return 1;
  }
  for (int j = 0; j < M * ndata; ++j)
    cm[j] = sin((FLT)1.3 * j) + IMA * cos((FLT)0.9 * j); // reset cm for 2d1many
  ier = FINUFFT2D1MANY(0, M, x, x, cm, +1, 0, N, N, Fm, &opts);
  if (ier != FINUFFT_ERR_NTRANS_NOTVALID) {
    printf("2d1many ndata=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT2D1MANY(ndata, M, x, x, cm, +1, 0, N, N, Fm, &opts);
  if (ier != FINUFFT_WARN_EPS_TOO_SMALL) {
    printf("2d1many tol=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT2D1MANY(ndata, M, x, x, cm, +1, acc, 0, 0, Fm, &opts);
  if (ier) {
    printf("2d1many Ns=Nt=0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT2D1MANY(ndata, M, x, x, cm, +1, acc, 0, N, Fm, &opts);
  if (ier) {
    printf("2d1many Ns=0,Nt>0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT2D1MANY(ndata, M, x, x, cm, +1, acc, N, 0, Fm, &opts);
  if (ier) {
    printf("2d1many Ns>0,Nt=0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT2D1MANY(ndata, 0, x, x, cm, +1, acc, N, N, Fm, &opts);
  t   = twonorm(N * ndata, Fm);
  if (ier || t != 0.0) {
    printf("2d1many M=0:\tier=%d nrm(Fm)=%.3g\n", ier, t);
    return 1;
  }
  for (int k = 0; k < NN * ndata; ++k)
    Fm[k] = sin((FLT)0.7 * k) + IMA * cos((FLT)0.3 * k); // reset Fm for t2
  ier = FINUFFT2D2MANY(0, M, x, x, cm, +1, 0, N, N, Fm, &opts);
  if (ier != FINUFFT_ERR_NTRANS_NOTVALID) {
    printf("2d2many ndata=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT2D2MANY(ndata, M, x, x, cm, +1, 0, N, N, Fm, &opts);
  if (ier != FINUFFT_WARN_EPS_TOO_SMALL) {
    printf("2d2many tol=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT2D2MANY(ndata, M, x, x, cm, +1, acc, 0, 0, Fm, &opts);
  t   = twonorm(M * ndata, cm);
  if (ier || t != 0.0) {
    printf("2d2many Ns=Nt=0:\tier=%d nrm(cm)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT2D2MANY(ndata, M, x, x, cm, +1, acc, 0, N, Fm, &opts);
  t   = twonorm(M * ndata, cm);
  if (ier || t != 0.0) {
    printf("2d2many Ns=0,Nt>0:\tier=%d nrm(cm)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT2D2MANY(ndata, M, x, x, cm, +1, acc, N, 0, Fm, &opts);
  t   = twonorm(M * ndata, cm);
  if (ier || t != 0.0) {
    printf("2d2many Ns>0,Nt=0:\tier=%d nrm(cm)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT2D2MANY(ndata, 0, x, x, cm, +1, acc, N, N, Fm, &opts);
  if (ier) {
    printf("2d2many M=0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT2D3MANY(0, M, x, x, cm, +1, 0, N, s, s, Fm, &opts);
  if (ier != FINUFFT_ERR_NTRANS_NOTVALID) {
    printf("2d3many ndata=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT2D3MANY(ndata, M, x, x, cm, +1, 0, N, s, s, Fm, &opts);
  if (ier != FINUFFT_WARN_EPS_TOO_SMALL) {
    printf("2d3many tol=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT2D3MANY(ndata, M, x, x, cm, +1, acc, 0, s, s, Fm, &opts);
  if (ier) {
    printf("2d3many nk=0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT2D3MANY(ndata, 0, x, x, cm, +1, acc, N, s, s, Fm, &opts);
  t   = twonorm(N, Fm);
  if (ier || t != 0.0) {
    printf("2d3many M=0:\tier=%d nrm(F)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT2D3MANY(ndata, 1, x, x, cm, +1, acc, N, s, s, Fm, &opts); // XK prod
                                                                         // formally 0
  // we don't check the M=nk=1 case for >1D since guess that 1D would catch it.
  if (ier) {
    printf("2d3many M=nk=1:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT2D3MANY(ndata, M, x, x, cm, +1, acc, N, shuge, shuge, Fm, &opts);
  if (ier == 0) { // any nonzero code accepted here
    printf("2d3many XK prod too big:\twrong error code %d\n", ier);
    return 1;
  }

  // 3333333333333333333333333333333333333333333333333333333333333333333333333
  printf("3D dumb cases.\n"); // z=y=x, and u=t=s in type 3
  ier = FINUFFT3D1(M, x, x, x, c, +1, 0, N, N, N, F, &opts);
  if (ier != FINUFFT_WARN_EPS_TOO_SMALL) {
    printf("3d1 tol=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT3D1(M, x, x, x, c, +1, acc, 0, 0, 0, F, &opts);
  if (ier) {
    printf("3d1 Ns=Nt=Nu=0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT3D1(M, x, x, x, c, +1, acc, 0, N, 0, F, &opts);
  if (ier) {
    printf("3d1 Ns=0,Nt>0,Nu=0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT3D1(M, x, x, x, c, +1, acc, N, 0, N, F, &opts);
  if (ier) {
    printf("3d1 Ns>0,Nt=0,Nu>0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT3D1(0, x, x, x, c, +1, acc, N, N, N, F, &opts);
  t   = twonorm(N, F);
  if (ier || t != 0.0) {
    printf("3d1 M=0:\tier=%d nrm(F)=%.3g\n", ier, t);
    return 1;
  }
  for (int k = 0; k < NN; ++k)
    F[k] = sin((FLT)0.8 * k) - IMA * cos((FLT)0.3 * k); // set F for t2
  ier = FINUFFT3D2(M, x, x, x, c, +1, 0, N, N, N, F, &opts);
  if (ier != FINUFFT_WARN_EPS_TOO_SMALL) {
    printf("3d2 tol=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT3D2(M, x, x, x, c, +1, acc, 0, 0, 0, F, &opts);
  t   = twonorm(M, c);
  if (ier || t != 0.0) {
    printf("3d2 Ns=Nt=Nu=0:\tier=%d nrm(c)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT3D2(M, x, x, x, c, +1, acc, N, 0, 0, F, &opts);
  t   = twonorm(M, c);
  if (ier || t != 0.0) {
    printf("3d2 Ns>0,Nt=Nu=0:\tier=%d nrm(c)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT3D2(M, x, x, x, c, +1, acc, 0, N, 0, F, &opts);
  t   = twonorm(M, c);
  if (ier || t != 0.0) {
    printf("3d2 Ns=0,Nt>0,Nu=0:\tier=%d nrm(c)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT3D2(M, x, x, x, c, +1, acc, 0, 0, N, F, &opts);
  t   = twonorm(M, c);
  if (ier || t != 0.0) {
    printf("3d2 Ns=Nt=0,Nu>0:\tier=%d nrm(c)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT3D2(0, x, x, x, c, +1, acc, N, N, N, F, &opts);
  if (ier) {
    printf("3d2 M=0:\tier=%d\n", ier);
    return ier;
  }
  for (int j = 0; j < M; ++j)
    c[j] = sin((FLT)1.2 * j) - IMA * cos((FLT)0.8 * j); // reset c for t3
  ier = FINUFFT3D3(M, x, x, x, c, +1, 0, N, s, s, s, F, &opts);
  if (ier != FINUFFT_WARN_EPS_TOO_SMALL) {
    printf("3d3 tol=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT3D3(M, x, x, x, c, +1, acc, 0, s, s, s, F, &opts);
  if (ier) {
    printf("3d3 nk=0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT3D3(0, x, x, x, c, +1, acc, N, s, s, s, F, &opts);
  t   = twonorm(N, F);
  if (ier || t != 0.0) {
    printf("3d3 M=0:\tier=%d nrm(F)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT3D3(1, x, x, x, c, +1, acc, N, s, s, s, F, &opts); // XK prod formally 0
  // we don't check the M=nk=1 case for >1D since guess that 1D would catch it.
  if (ier) {
    printf("3d3 M=nk=1:\tier=%d\n", ier);
    return ier;
  }
  for (int k = 0; k < N; ++k) shuge[k] = pow(huge, 1. / 3) * s[k]; // less huge coords
  ier = FINUFFT3D3(M, x, x, x, c, +1, acc, N, shuge, shuge, shuge, F, &opts);
  if (ier == 0) { // any nonzero code accepted here
    printf("3d3 XK prod too big:\twrong error code %d\n", ier);
    return 1;
  }
  for (int j = 0; j < M * ndata; ++j)
    cm[j] = sin(-(FLT)1.2 * j) + IMA * cos((FLT)1.1 * j); // reset cm for 3d1many
  ier = FINUFFT3D1MANY(0, M, x, x, x, cm, +1, 0, N, N, N, Fm, &opts);
  if (ier != FINUFFT_ERR_NTRANS_NOTVALID) {
    printf("3d1many ndata=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT3D1MANY(ndata, M, x, x, x, cm, +1, 0, N, N, N, Fm, &opts);
  if (ier != FINUFFT_WARN_EPS_TOO_SMALL) {
    printf("3d1many tol=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT3D1MANY(ndata, M, x, x, x, cm, +1, acc, 0, 0, 0, Fm, &opts);
  if (ier) {
    printf("3d1many Ns=Nt=Nu=0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT3D1MANY(ndata, M, x, x, x, cm, +1, acc, N, 0, 0, Fm, &opts);
  if (ier) {
    printf("3d1many Ns>0,Nt=Nu=0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT3D1MANY(ndata, M, x, x, x, cm, +1, acc, 0, N, 0, Fm, &opts);
  if (ier) {
    printf("3d1many Ns=0,Nt>0,Nu=0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT3D1MANY(ndata, M, x, x, x, cm, +1, acc, 0, 0, N, Fm, &opts);
  if (ier) {
    printf("3d1many Ns=Nt=0,Nu>0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT3D1MANY(ndata, 0, x, x, x, cm, +1, acc, N, N, N, Fm, &opts);
  t   = twonorm(N * ndata, Fm);
  if (ier || t != 0.0) {
    printf("3d1many M=0:\tier=%d nrm(Fm)=%.3g\n", ier, t);
    return 1;
  }
  for (int k = 0; k < NN * ndata; ++k)
    Fm[k] = sin((FLT)0.6 * k) - IMA * cos((FLT)0.3 * k); // reset Fm for t2
  ier = FINUFFT3D2MANY(0, M, x, x, x, cm, +1, 0, N, N, N, Fm, &opts);
  if (ier != FINUFFT_ERR_NTRANS_NOTVALID) {
    printf("3d2many ndata=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT3D2MANY(ndata, M, x, x, x, cm, +1, 0, N, N, N, Fm, &opts);
  if (ier != FINUFFT_WARN_EPS_TOO_SMALL) {
    printf("3d2many tol=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT3D2MANY(ndata, M, x, x, x, cm, +1, acc, 0, 0, 0, Fm, &opts);
  t   = twonorm(M * ndata, cm);
  if (ier || t != 0.0) {
    printf("3d2many Ns=Nt=Nu=0:\tier=%d nrm(cm)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT3D2MANY(ndata, M, x, x, x, cm, +1, acc, N, 0, 0, Fm, &opts);
  t   = twonorm(M * ndata, cm);
  if (ier || t != 0.0) {
    printf("3d2many Ns>0,Nt=Nu=0:\tier=%d nrm(cm)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT3D2MANY(ndata, M, x, x, x, cm, +1, acc, 0, N, 0, Fm, &opts);
  t   = twonorm(M * ndata, cm);
  if (ier || t != 0.0) {
    printf("3d2many Ns=0,Nt>0,Nu=0:\tier=%d nrm(cm)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT3D2MANY(ndata, M, x, x, x, cm, +1, acc, 0, 0, N, Fm, &opts);
  t   = twonorm(M * ndata, cm);
  if (ier || t != 0.0) {
    printf("3d2many Ns=Nt=0,Nu>0:\tier=%d nrm(cm)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT3D2MANY(ndata, 0, x, x, x, cm, +1, acc, N, N, N, Fm, &opts);
  if (ier) {
    printf("3d2many M=0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT3D3MANY(0, M, x, x, x, cm, +1, 0, N, s, s, s, Fm, &opts);
  if (ier != FINUFFT_ERR_NTRANS_NOTVALID) {
    printf("3d3many ndata=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT3D3MANY(ndata, M, x, x, x, cm, +1, 0, N, s, s, s, Fm, &opts);
  if (ier != FINUFFT_WARN_EPS_TOO_SMALL) {
    printf("3d3many tol=0:\twrong err code %d\n", ier);
    return 1;
  }
  ier = FINUFFT3D3MANY(ndata, M, x, x, x, cm, +1, acc, 0, s, s, s, Fm, &opts);
  if (ier) {
    printf("3d3many nk=0:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT3D3MANY(ndata, 0, x, x, x, cm, +1, acc, N, s, s, s, Fm, &opts);
  t   = twonorm(N, Fm);
  if (ier || t != 0.0) {
    printf("3d3many M=0:\tier=%d nrm(F)=%.3g\n", ier, t);
    return 1;
  }
  ier = FINUFFT3D3MANY(ndata, 1, x, x, x, cm, +1, acc, N, s, s, s, Fm, &opts); // XK
                                                                               // prod
                                                                               // formally
                                                                               // 0
  // we don't check the M=nk=1 case for >1D since guess that 1D would catch it.
  if (ier) {
    printf("3d3many M=nk=1:\tier=%d\n", ier);
    return ier;
  }
  ier = FINUFFT3D3MANY(ndata, M, x, x, x, cm, +1, acc, N, shuge, shuge, shuge, Fm, &opts);
  if (ier == 0) { // any nonzero code accepted here
    printf("3d3many XK prod too big:\twrong error code %d\n", ier);
    return 1;
  }

  free(x);
  free(c);
  free(F);
  free(s);
  free(shuge);
  free(cm);
  free(Fm);
  free(Fe);

  // GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
  // some dumb tests for guru interface to induce free() crash in destroy...
  FINUFFT_PLAN plan;
  BIGINT Ns[1] = {0}; // since dim=1, don't have to make length 3
  FINUFFT_MAKEPLAN(1, 1, Ns, +1, 1, acc, &plan, NULL); // type 1, now kill it
  FINUFFT_DESTROY(plan);
  FINUFFT_MAKEPLAN(3, 1, Ns, +1, 1, acc, &plan, NULL); // type 3, now kill it
  FINUFFT_DESTROY(plan);
  // *** todo: more extensive bad inputs and error catching in guru...

#ifdef SINGLE
  printf("dumbinputsf passed.\n");
#else
  printf("dumbinputs passed.\n");
#endif

  return 0;
}
