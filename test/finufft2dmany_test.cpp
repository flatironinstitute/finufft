#include <finufft_old.h>
#include <finufft_legacy.h>
#include <fftw_defs.h>
#include <dirft.h>
#include <math.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>

// how big a problem to do full direct DFT check in 2D...
#define BIGPROB 1e8

// for omp rand filling
#define CHUNK 1000000

int main(int argc, char* argv[])
/* Test executable for finufft in 2d many interface, types 1,2.

   Usage: finufft2dmany_test [ntransf [Nmodes1 Nmodes2 [Nsrc [tol [debug [spread_sort [upsampfac]]]]]]]

   debug = 0: rel errors and overall timing, 1: timing breakdowns
           2: also spreading output

   Example: finufft2dmany_test 1000 1e2 1e2 1e4 1e-6 1 2 2.0

   Melody Shih Jun 2018, based on Barnett Feb 2017.
   Barnett removed many_seq for simplicity, 7/27/18.
*/
{
  BIGINT M = 1e6, N1 = 1000, N2 = 500;  // defaults: M = # srcs, N1,N2 = # modes
  int debug;
  int ntransf = 400;                      // # of vectors for "many" interface

  double w, tol = 1e-6;          // default
  double upsampfac = 2.0;        // default
  nufft_opts opts; finufft_default_opts(&opts);
  opts.debug = 0;            // 1 to see some timings
  // opts.fftw = FFTW_MEASURE;  // change from usual FFTW_ESTIMATE
  int isign = +1;             // choose which exponential sign to test
  if (argc>1) { sscanf(argv[1],"%lf",&w); ntransf = (int)w; }
  if (argc>2) {
    sscanf(argv[2],"%lf",&w); N1 = (BIGINT)w;
    sscanf(argv[3],"%lf",&w); N2 = (BIGINT)w;
  }
  if (argc>4) { sscanf(argv[4],"%lf",&w); M = (BIGINT)w; }
  if (argc>5) {
    sscanf(argv[5],"%lf",&tol);
    if (tol<=0.0) { printf("tol must be positive!\n"); return 1; }
  }
  if (argc>6) sscanf(argv[6],"%d",&debug);
  opts.debug = debug;
  opts.spread_debug = (debug>1) ? 1 : 0;  // see output from spreader
  if (argc>7) sscanf(argv[7],"%d",&opts.spread_sort);
  if (argc>8) sscanf(argv[8],"%lf",&upsampfac);
  opts.upsampfac=(FLT)upsampfac;

  if (argc==1 || argc==2 || argc>9) {
    fprintf(stderr,"Usage: finufft2d_test [ntransf [N1 N2 [Nsrc [tol [debug [spread_sort [upsampfac]]]]]]]\n");
    return 1;
  }
  cout << scientific << setprecision(15);
  BIGINT N = N1*N2;

  FLT* x = (FLT*)malloc(sizeof(FLT)*M);  // NU pts x coords
  FLT* y = (FLT*)malloc(sizeof(FLT)*M);  // NU pts y coords
  CPX* c = (CPX*)malloc(sizeof(CPX)*M*ntransf);   // strengths 
  CPX* F = (CPX*)malloc(sizeof(CPX)*N*ntransf);   // mode ampls

#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(dynamic,CHUNK)
    for (BIGINT j=0; j<M; ++j) {
      x[j] = M_PI*randm11r(&se);
      y[j] = M_PI*randm11r(&se);
    }
#pragma omp for schedule(dynamic,CHUNK)
    for (BIGINT j = 0; j<ntransf*M; ++j)
    {
        c[j] = crandm11r(&se);
    }
  }

  printf("test 2dmany type-1:\n"); // -------------- type 1
  CNTime timer; timer.start();
  int ier = finufft2d1many(ntransf,M,x,y,c,isign,tol,N1,N2,F,opts);
  double ti=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("    %d of: %lld NU pts to (%lld,%lld) modes in %.3g s or  \t%.3g NU pts/s\n", ntransf,(long long)M,(long long)N1,(long long)N2,ti,ntransf*M/ti);

  // compare the result with finufft2d1
  FFTW_FORGET_WISDOM(); // for fair comparison
  CPX* cstart;
  CPX* Fstart;
  CPX* F_finufft2d1 = (CPX*)malloc(sizeof(CPX)*N*ntransf);
  FLT maxerror = 0.0;
  opts.debug = 0;       // don't output timing for calls of finufft2d1
  opts.spread_debug = 0;
  timer.restart();
  for (int k= 0; k<ntransf; ++k)
  {
    cstart = c+k*M;
    Fstart = F_finufft2d1+k*N;
    ier = finufft2d1_old(M,x,y,cstart,isign,tol,N1,N2,Fstart,opts);
  }
  double t=timer.elapsedsec();
  printf("\tspeedup (T_finufft2d1 / T_finufft2d1many) = %.3g\n", t/ti);

  int d = floor(ntransf/2);    // choose a data to check
  BIGINT nt1 = (BIGINT)(0.37*N1), nt2 = (BIGINT)(0.26*N2);  // choose some mode index to check
  CPX Ft = CPX(0,0), J = IMA*(FLT)isign;
  for (BIGINT j=0; j<M; ++j)
    Ft += c[j+d*M] * exp(J*(nt1*x[j]+nt2*y[j]));   // crude direct
  BIGINT it = N1/2+nt1 + N1*(N2/2+nt2);   // index in complex F as 1d array
  printf("one mode: rel err in F[%lld,%lld] of data[%d] is %.3g\n",(long long)nt1,(long long)nt2,d,abs(Ft-F[it+d*N])/infnorm(N,F+d*N));

  // Check accuracy (worst over the ntransf)
  for (int k = 0; k < ntransf; ++k)
    maxerror = max(maxerror, relerrtwonorm(N,F_finufft2d1+k*N,F+k*N));
  printf("err check vs non-many: sup ( ||F_many-F||_2 / ||F||_2  ) =  %.3g\n",maxerror);
  free(F_finufft2d1);

  FFTW_FORGET_WISDOM();
  printf("test 2dmany type-2:\n"); // -------------- type 2

#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(dynamic,CHUNK)
    for (BIGINT m=0; m<N*ntransf; ++m) F[m] = crandm11r(&se);
  }

  opts.debug = debug; // set debug flags back to original setting
  opts.spread_debug = (debug>1) ? 1 : 0;
  timer.restart();
  ier = finufft2d2many(ntransf,M,x,y,c,isign,tol,N1,N2,F,opts);
  ti=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("    %d of: (%lld,%lld) modes to %lld NU pts in %.3g s \t%.3g NU pts/s\n", ntransf,(long long)N1,(long long)N2,(long long)M,ti,ntransf*M/ti);
  
  FFTW_FORGET_WISDOM();
  opts.debug = 0;        // don't output timing for calls of finufft2d2
  opts.spread_debug = 0;
  
  // compare the result with finufft2d2...
  CPX* c_finufft2d2 = (CPX*)malloc(sizeof(CPX)*M*ntransf);
  timer.restart();
  for (int k=0; k<ntransf; ++k)
  {
    cstart = c_finufft2d2+k*M;
    Fstart = F+k*N;
    ier = finufft2d2_old(M,x,y,cstart,isign,tol,N1,N2,Fstart,opts);
  }
  t = timer.elapsedsec();
  printf("\tspeedup (T_finufft2d2 / T_finufft2d2many) = %.3g\n", t/ti);
  
  d = floor(ntransf/2); // choose a data to check
  BIGINT jt = M/2;    // check arbitrary choice of one targ pt
  CPX ct = CPX(0,0);
  BIGINT m=0;
  for (BIGINT m2=-(N2/2); m2<=(N2-1)/2; ++m2)  // loop in correct order over F
    for (BIGINT m1=-(N1/2); m1<=(N1-1)/2; ++m1)
      ct += F[d*N + m++] * exp(J*(m1*x[jt] + m2*y[jt]));   // crude direct
  printf("one targ: rel err in c[%lld] of data[%d] is %.3g\n",(long long)jt,d,abs(ct-c[jt+d*M])/infnorm(M,c+d*M));

  maxerror = 0.0;           // worst error over the ntransf
  for (int k = 0; k < ntransf; ++k)
    maxerror = max(maxerror, relerrtwonorm(M,c_finufft2d2+k*M,c+k*M));
  printf("err check vs non-many: sup ( ||c_many-c||_2 / ||c||_2 ) =  %.3g\n",maxerror);
  free(c_finufft2d2);

  

  FFTW_FORGET_WISDOM();
  printf("test 2dmany type-3:\n"); // -------------- type 3

  FLT* s_freq = (FLT*)malloc(sizeof(FLT)*N);    // targ freqs (1-cmpt)
  FLT* t_freq = (FLT*)malloc(sizeof(FLT)*N);    // targ freqs (2-cmpt)
  FLT S1 = (FLT)N1/2;                   // choose freq range sim to type 1
  FLT S2 = (FLT)N2/2;

#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(dynamic,CHUNK)
    for (BIGINT k=0; k<N; ++k) {
      s_freq[k] = S1*(1.7 + randm11r(&se));    //S*(1.7 + k/(FLT)N); // offset the freqs
      t_freq[k] = S2*(-0.5 + randm11r(&se));
    }
  }

  timer.restart();
  ier = finufft2d3many(ntransf,M,x,y,c,isign,tol,N,s_freq,t_freq,F,opts);
  ti=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("    %d of: %lld NU to %lld NU in %.3g s   %.3g srcs/s, %.3g targs/s\n",ntransf, (long long)M,(long long)N,ti,M/ti,N/ti);


  d = floor(ntransf/2); // choose a data to check
  BIGINT kt = N/2;          // check arbitrary choice of one targ pt
  Ft = CPX(0,0);
  for (BIGINT j=0;j<M;++j)
    Ft += c[d*M + j] * exp(IMA*(FLT)isign*(s_freq[kt]*x[j] + t_freq[kt]*y[j]));
  printf("one targ: rel err in F[%lld] is %.3g\n",(long long)kt,abs(Ft-F[kt+d*N])/infnorm(N,F+d*N));


// compare the result with finufft2d3...
  CPX* f_finufft2d3 = (CPX*)malloc(sizeof(CPX)*N*ntransf);
  timer.restart();
  for (int k=0; k<ntransf; ++k)
  {
    Fstart = f_finufft2d3+k*N;
    cstart = c+k*M;
    ier = finufft2d3_old(M,x,y,cstart,isign,tol,N, s_freq,t_freq,Fstart,opts);
  }
  t = timer.elapsedsec();
  printf("\tspeedup (T_finufft2d3 / T_finufft2d3many) = %.3g\n", t/ti);
  
  maxerror = 0.0;           // worst error over the ntransf
  for (int k = 0; k < ntransf; ++k)
    maxerror = max(maxerror, relerrtwonorm(M,f_finufft2d3+d*N,F+d*N));
  printf("err check vs non-many: sup ( ||c_many-c||_2 / ||c||_2 ) =  %.3g\n",maxerror);
  free(f_finufft2d3);
  
  
  free(x); free(y); free(c); free(F);
  return ier;
}
