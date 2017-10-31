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

   Usage: finufft2d_test [Nmodes1 Nmodes2 [Nsrc [tol [debug [spread_sort]]]]]

   debug = 0: rel errors and overall timing, 1: timing breakdowns
           2: also spreading output

   Example: finufft2d_test 1000 1000 1000000 1e-12

   Barnett 2/1/17
*/
{
  INT M = 1e6, N1 = 1000, N2 = 500;  // defaults: M = # srcs, N1,N2 = # modes
  double w, tol = 1e-6;          // default
  nufft_opts opts;
  finufft_default_opts(opts);
  opts.debug = 0;            // 1 to see some timings
  // opts.fftw = FFTW_MEASURE;  // change from usual FFTW_ESTIMATE
  int isign = +1;             // choose which exponential sign to test
  if (argc>1) {
    sscanf(argv[1],"%lf",&w); N1 = (INT)w;
    sscanf(argv[2],"%lf",&w); N2 = (INT)w;
  }
  if (argc>3) { sscanf(argv[3],"%lf",&w); M = (INT)w; }
  if (argc>4) {
    sscanf(argv[4],"%lf",&tol);
    if (tol<=0.0) { printf("tol must be positive!\n"); return 1; }
  }
  if (argc>5) sscanf(argv[5],"%d",&opts.debug);
  opts.spread_debug = (opts.debug>1) ? 1 : 0;  // see output from spreader
  if (argc>6) sscanf(argv[6],"%d",&opts.spread_sort);
  if (argc==1 || argc==2 || argc>7) {
    fprintf(stderr,"Usage: finufft2d_test [N1 N2 [Nsrc [tol [debug [spread_sort]]]]]\n");
    return 1;
  }
  cout << scientific << setprecision(15);
  INT N = N1*N2;

  FLT *x = (FLT *)malloc(sizeof(FLT)*M);        // NU pts x coords
  FLT *y = (FLT *)malloc(sizeof(FLT)*M);        // NU pts y coords
  CPX* c = (CPX*)malloc(sizeof(CPX)*M);   // strengths 
  CPX* F = (CPX*)malloc(sizeof(CPX)*N);   // mode ampls
#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();  // needed for parallel random #s
#pragma omp for schedule(dynamic,CHUNK)
    for (INT j=0; j<M; ++j) {
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
    printf("\t%ld NU pts to (%ld,%ld) modes in %.3g s \t%.3g NU pts/s\n",
	   (INT64)M,(INT64)N1,(INT64)N2,ti,M/ti);

  INT nt1 = (INT)(0.37*N1), nt2 = (INT)(0.26*N2);  // choose some mode index to check
  CPX Ft = (0,0), J = ima*(FLT)isign;
  for (INT j=0; j<M; ++j)
    Ft += c[j] * exp(J*(nt1*x[j]+nt2*y[j]));   // crude direct
  INT it = N1/2+nt1 + N1*(N2/2+nt2);   // index in complex F as 1d array
  printf("one mode: rel err in F[%ld,%ld] is %.3g\n",(INT64)nt1,(INT64)nt2,abs(Ft-F[it])/infnorm(N,F));
  if ((INT64)M*N<=BIGPROB) {                   // also check vs full direct eval
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
    for (INT m=0; m<N; ++m) F[m] = crandm11r(&se);
  }
  timer.restart();
  ier = finufft2d2(M,x,y,c,isign,tol,N1,N2,F,opts);
  ti=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("\t(%ld,%ld) modes to %ld NU pts in %.3g s \t%.3g NU pts/s\n",(INT64)N1,(INT64)N2,(INT64)M,ti,M/ti);

  INT jt = M/2;          // check arbitrary choice of one targ pt
  CPX ct = (0,0);
  INT m=0;
  for (INT m2=-(N2/2); m2<=(N2-1)/2; ++m2)  // loop in correct order over F
    for (INT m1=-(N1/2); m1<=(N1-1)/2; ++m1)
      ct += F[m++] * exp(J*(m1*x[jt] + m2*y[jt]));   // crude direct
  printf("one targ: rel err in c[%ld] is %.3g\n",(INT64)jt,abs(ct-c[jt])/infnorm(M,c));
  if ((INT64)M*N<=BIGPROB) {                  // also full direct eval
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
    for (INT j=0; j<M; ++j) {
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
    for (INT k=0; k<N; ++k) {
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
    printf("\t%ld NU to %ld NU in %.3g s   %.3g srcs/s, %.3g targs/s\n",(INT64)M,(INT64)N,ti,M/ti,N/ti);

  INT kt = N/2;          // check arbitrary choice of one targ pt
  Ft = (0,0);
  for (INT j=0;j<M;++j)
    Ft += c[j] * exp(ima*(FLT)isign*(s[kt]*x[j] + t[kt]*y[j]));
  printf("one targ: rel err in F[%ld] is %.3g\n",(INT64)kt,abs(Ft-F[kt])/infnorm(N,F));
  if ((INT64)M*N<=BIGPROB) {                  // also full direct eval
    CPX* Ft = (CPX*)malloc(sizeof(CPX)*N);
    dirft2d3(M,x,y,c,isign,N,s,t,Ft);       // writes to F
    printf("dirft2d: rel l2-err of result F is %.3g\n",relerrtwonorm(N,Ft,F));
    //cout<<"s t, F, Ft, F/Ft:\n"; for (int k=0;k<N;++k) cout<<s[k]<<" "<<t[k]<<", "<<F[k]<<",\t"<<Ft[k]<<",\t"<<F[k]/Ft[k]<<endl;
    free(Ft);
  }

  free(x); free(y); free(c); free(F); free(s); free(t);
  return ier;
}
