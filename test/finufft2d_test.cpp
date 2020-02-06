#include "../src/finufft.h"
#include "../src/dirft.h"
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
/* Test executable for finufft in 2d, all 3 types

   Usage: finufft2d_test [Nmodes1 Nmodes2 [Nsrc [tol [debug [spread_sort [upsampfac]]]]]]

   debug = 0: rel errors and overall timing, 1: timing breakdowns
           2: also spreading output

   Example: finufft2d_test 1000 1000 1000000 1e-12 1 2 2.0

   Barnett 2/1/17
*/
{
  BIGINT M = 1e6, N1 = 1000, N2 = 500;  // defaults: M = # srcs, N1,N2 = # modes
  double w, tol = 1e-6;          // default
  double upsampfac = 2.0;        // default
  nufft_opts opts; finufft_default_opts(&opts);
  opts.debug = 0;            // 1 to see some timings
  // opts.fftw = FFTW_MEASURE;  // change from usual FFTW_ESTIMATE
  int isign = +1;             // choose which exponential sign to test
  if (argc>1) {
    sscanf(argv[1],"%lf",&w); N1 = (BIGINT)w;
    sscanf(argv[2],"%lf",&w); N2 = (BIGINT)w;
  }
  if (argc>3) { sscanf(argv[3],"%lf",&w); M = (BIGINT)w; }
  if (argc>4) {
    sscanf(argv[4],"%lf",&tol);
    if (tol<=0.0) { printf("tol must be positive!\n"); return 1; }
  }
  if (argc>5) sscanf(argv[5],"%d",&opts.debug);
  opts.spread_debug = (opts.debug>1) ? 1 : 0;  // see output from spreader
  if (argc>6) sscanf(argv[6],"%d",&opts.spread_sort);
  if (argc>7) sscanf(argv[7],"%lf",&upsampfac);
  opts.upsampfac=(FLT)upsampfac;
  if (argc==1 || argc==2 || argc>8) {
    fprintf(stderr,"Usage: finufft2d_test [N1 N2 [Nsrc [tol [debug [spread_sort [upsampfac]]]]]]\n");
    return 1;
  }
  cout << scientific << setprecision(15);
  BIGINT N = N1*N2;

  FLT *x = (FLT *)malloc(sizeof(FLT)*M);        // NU pts x coords
  FLT *y = (FLT *)malloc(sizeof(FLT)*M);        // NU pts y coords
  CPX* c = (CPX*)malloc(sizeof(CPX)*M);   // strengths 
  CPX* F = (CPX*)malloc(sizeof(CPX)*N);   // mode ampls
#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();  // needed for parallel random #s
#pragma omp for schedule(dynamic,CHUNK)
    for (BIGINT j=0; j<M; ++j) {
      x[j] = M_PI*randm11r(&se);
      y[j] = M_PI*randm11r(&se);
      c[j] = crandm11r(&se);
    }
  }

  printf("test 2d type-1:\n"); // -------------- type 1
  CNTime timer; timer.start();
  int ier = finufft2d1(M,x,y,c,isign,tol,N1,N2,F,opts);
  double ti=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("\t%lld NU pts to (%lld,%lld) modes in %.3g s \t%.3g NU pts/s\n",
	   (long long)M,(long long)N1,(long long)N2,ti,M/ti);

  BIGINT nt1 = (BIGINT)(0.37*N1), nt2 = (BIGINT)(0.26*N2);  // choose some mode index to check
  CPX Ft = CPX(0,0), J = IMA*(FLT)isign;
  for (BIGINT j=0; j<M; ++j)
    Ft += c[j] * exp(J*(nt1*x[j]+nt2*y[j]));   // crude direct
  BIGINT it = N1/2+nt1 + N1*(N2/2+nt2);   // index in complex F as 1d array
  printf("one mode: rel err in F[%lld,%lld] is %.3g\n",(long long)nt1,(long long)nt2,abs(Ft-F[it])/infnorm(N,F));
  if ((int64_t)M*N<=BIGPROB) {                   // also check vs full direct eval
    CPX* Ft = (CPX*)malloc(sizeof(CPX)*N);
    dirft2d1(M,x,y,c,isign,N1,N2,Ft);
    printf("dirft2d: rel l2-err of result F is %.3g\n",relerrtwonorm(N,Ft,F));
    free(Ft);
  }

  printf("test 2d type-2:\n"); // -------------- type 2
#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(dynamic,CHUNK)
    for (BIGINT m=0; m<N; ++m) F[m] = crandm11r(&se);
  }
  timer.restart();
  ier = finufft2d2(M,x,y,c,isign,tol,N1,N2,F,opts);
  ti=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("\t(%lld,%lld) modes to %lld NU pts in %.3g s \t%.3g NU pts/s\n",(long long)N1,(long long)N2,(long long)M,ti,M/ti);

  BIGINT jt = M/2;          // check arbitrary choice of one targ pt
  CPX ct = CPX(0,0);
  BIGINT m=0;
  for (BIGINT m2=-(N2/2); m2<=(N2-1)/2; ++m2)  // loop in correct order over F
    for (BIGINT m1=-(N1/2); m1<=(N1-1)/2; ++m1)
      ct += F[m++] * exp(J*(m1*x[jt] + m2*y[jt]));   // crude direct
  printf("one targ: rel err in c[%lld] is %.3g\n",(long long)jt,abs(ct-c[jt])/infnorm(M,c));
  if ((int64_t)M*N<=BIGPROB) {                  // also full direct eval
    CPX* ct = (CPX*)malloc(sizeof(CPX)*M);
    dirft2d2(M,x,y,ct,isign,N1,N2,F);
    printf("dirft2d: rel l2-err of result c is %.3g\n",relerrtwonorm(M,ct,c));
    //cout<<"c,ct:\n"; for (int j=0;j<M;++j) cout<<c[j]<<"\t"<<ct[j]<<endl;
    free(ct);
  }

  printf("test 2d type-3:\n"); // -------------- type 3
  // reuse the strengths c, interpret N as number of targs:
#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(dynamic,CHUNK)
    for (BIGINT j=0; j<M; ++j) {
      x[j] = 2.0 + M_PI*randm11r(&se);      // new x_j srcs, offset from origin
      y[j] = -3.0 + M_PI*randm11r(&se);     // " y_j
    }
  }
  FLT* s = (FLT*)malloc(sizeof(FLT)*N);    // targ freqs (1-cmpt)
  FLT* t = (FLT*)malloc(sizeof(FLT)*N);    // targ freqs (2-cmpt)
  FLT S1 = (FLT)N1/2;                   // choose freq range sim to type 1
  FLT S2 = (FLT)N2/2;
#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(dynamic,CHUNK)
    for (BIGINT k=0; k<N; ++k) {
      s[k] = S1*(1.7 + randm11r(&se));    //S*(1.7 + k/(FLT)N); // offset the freqs
      t[k] = S2*(-0.5 + randm11r(&se));
    }
  }
  timer.restart();
  ier = finufft2d3(M,x,y,c,isign,tol,N,s,t,F,opts);
  ti=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("\t%lld NU to %lld NU in %.3g s         \t%.3g tot NU pts/s\n",(long long)M,(long long)N,ti,(M+N)/ti);

  BIGINT kt = N/2;          // check arbitrary choice of one targ pt
  Ft = CPX(0,0);
  for (BIGINT j=0;j<M;++j)
    Ft += c[j] * exp(IMA*(FLT)isign*(s[kt]*x[j] + t[kt]*y[j]));
  printf("one targ: rel err in F[%lld] is %.3g\n",(long long)kt,abs(Ft-F[kt])/infnorm(N,F));
  if (((int64_t)M)*N<=BIGPROB) {                  // also full direct eval
    CPX* Ft = (CPX*)malloc(sizeof(CPX)*N);
    dirft2d3(M,x,y,c,isign,N,s,t,Ft);       // writes to F
    printf("dirft2d: rel l2-err of result F is %.3g\n",relerrtwonorm(N,Ft,F));
    //cout<<"s t, F, Ft, F/Ft:\n"; for (int k=0;k<N;++k) cout<<s[k]<<" "<<t[k]<<", "<<F[k]<<",\t"<<Ft[k]<<",\t"<<F[k]/Ft[k]<<endl;
    free(Ft);
  }

  free(x); free(y); free(c); free(F); free(s); free(t);
  return ier;
}
