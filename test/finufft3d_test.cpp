#include "../src/finufft.h"
#include "../src/dirft.h"
#include <math.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>

// how big a problem to do full direct DFT check in 3D...
#define BIGPROB 1e8

// for omp rand filling
#define CHUNK 1000000

int main(int argc, char* argv[])
/* Test executable for finufft in 3d, all 3 types.

   Usage: finufft3d_test [Nmodes1 Nmodes2 Nmodes3 [Nsrc [tol [debug [spread_sort]]]]]

   debug = 0: rel errors and overall timing, 1: timing breakdowns
           2: also spreading output

   Example: finufft3d_test 100 200 50 1e6 1e-12

   Barnett 2/2/17
*/
{
  INT M = 1e6, N1 = 100, N2 = 200, N3 = 50;  // defaults: M = # srcs, N1,N2,N3 = # modes
  double w, tol = 1e-6;       // default
  nufft_opts opts;
  finufft_default_opts(opts);
  opts.debug = 0;             // 1 to see some timings
  // opts.fftw = FFTW_MEASURE;  // change from usual FFTW_ESTIMATE
  int isign = +1;             // choose which exponential sign to test
  if (argc>1) {
    sscanf(argv[1],"%lf",&w); N1 = (INT)w;
    sscanf(argv[2],"%lf",&w); N2 = (INT)w;
    sscanf(argv[3],"%lf",&w); N3 = (INT)w;
  }
  if (argc>4) { sscanf(argv[4],"%lf",&w); M = (INT)w; }
  if (argc>5) {
    sscanf(argv[5],"%lf",&tol);
    if (tol<=0.0) { printf("tol must be positive!\n"); return 1; }
  }
  if (argc>6) sscanf(argv[6],"%d",&opts.debug);  // can be 0,1 or 2
  opts.spread_debug = (opts.debug>1) ? 1 : 0;  // see output from spreader
  if (argc>7) sscanf(argv[7],"%d",&opts.spread_sort);
   if (argc==1 || argc==2 || argc==3 || argc>8) {
    fprintf(stderr,"Usage: finufft3d_test [N1 N2 N3 [Nsrc [tol [debug [spread_sort]]]]]\n");
    return 1;
  }
  cout << scientific << setprecision(15);
  INT N = N1*N2*N3;

  FLT *x = (FLT *)malloc(sizeof(FLT)*M);        // NU pts x coords
  FLT *y = (FLT *)malloc(sizeof(FLT)*M);        // NU pts y coords
  FLT *z = (FLT *)malloc(sizeof(FLT)*M);        // NU pts z coords
  CPX* c = (CPX*)malloc(sizeof(CPX)*M);   // strengths 
  CPX* F = (CPX*)malloc(sizeof(CPX)*N);   // mode ampls
#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();  // needed for parallel random #s
#pragma omp for schedule(dynamic,CHUNK)
    for (INT j=0; j<M; ++j) {
      x[j] = M_PI*randm11r(&se);
      y[j] = M_PI*randm11r(&se);
      z[j] = M_PI*randm11r(&se);
      c[j] = crandm11r(&se);
    }
  }

  printf("test 3d type-1:\n"); // -------------- type 1
  CNTime timer; timer.start();
  int ier = finufft3d1(M,x,y,z,c,isign,tol,N1,N2,N3,F,opts);
  double ti=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
    exit(ier);
  } else
    printf("     %ld NU pts to (%ld,%ld,%ld) modes in %.3g s \t%.3g NU pts/s\n",
	   (INT64)M,(INT64)N1,(INT64)N2,(INT64)N3,ti,M/ti);

  INT nt1 = (INT)(0.37*N1), nt2 = (INT)(0.26*N2), nt3 = (INT)(-0.39*N3);  // choose mode to check
  CPX Ft = (0,0), J = ima*(FLT)isign;
  for (INT j=0; j<M; ++j)
    Ft += c[j] * exp(J*(nt1*x[j]+nt2*y[j]+nt3*z[j]));   // crude direct
  // index in complex F as 1d array...
  INT it = N1/2+nt1 + N1*(N2/2+nt2) + N1*N2*(N3/2+nt3);
  printf("one mode: rel err in F[%ld,%ld,%ld] is %.3g\n",(INT64)nt1,(INT64)nt2,(INT64)nt3,
	 abs(Ft-F[it])/infnorm(N,F));
  if ((INT64)M*N<=BIGPROB) {                   // also check vs full direct eval
    CPX* Ft = (CPX*)malloc(sizeof(CPX)*N);
    dirft3d1(M,x,y,z,c,isign,N1,N2,N3,Ft);
    printf("dirft3d: rel l2-err of result F is %.3g\n",relerrtwonorm(N,Ft,F));
    free(Ft);
  }
  
  printf("test 3d type-2:\n"); // -------------- type 2
#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(dynamic,CHUNK)
    for (INT m=0; m<N; ++m) F[m] = crandm11r(&se);
  }
  timer.restart();
  ier = finufft3d2(M,x,y,z,c,isign,tol,N1,N2,N3,F,opts);
  ti=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
    exit(ier);
  } else
    printf("     (%ld,%ld,%ld) modes to %ld NU pts in %.3g s \t%.3g NU pts/s\n",
	   (INT64)N1,(INT64)N2,(INT64)N3,(INT64)M,ti,M/ti);

  INT jt = M/2;          // check arbitrary choice of one targ pt
  CPX ct = (0,0);
  INT m=0;
  for (INT m3=-(N3/2); m3<=(N3-1)/2; ++m3)   // loop in F order
    for (INT m2=-(N2/2); m2<=(N2-1)/2; ++m2)
      for (INT m1=-(N1/2); m1<=(N1-1)/2; ++m1)
	ct += F[m++] * exp(J*(m1*x[jt] + m2*y[jt] + m3*z[jt]));   // direct
  printf("one targ: rel err in c[%ld] is %.3g\n",(INT64)jt,abs(ct-c[jt])/infnorm(M,c));
  if ((INT64)M*N<=BIGPROB) {                  // also full direct eval
    CPX* ct = (CPX*)malloc(sizeof(CPX)*M);
    dirft3d2(M,x,y,z,ct,isign,N1,N2,N3,F);
    printf("dirft3d: rel l2-err of result c is %.3g\n",relerrtwonorm(M,ct,c));
    free(ct);
  }

  printf("test 3d type-3:\n"); // -------------- type 3
  // reuse the strengths c, interpret N as number of targs:
#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(dynamic,CHUNK)
    for (INT j=0; j<M; ++j) {
      x[j] = 2.0 + M_PI*randm11r(&se);      // new x_j srcs, offset from origin
      y[j] = -3.0 + M_PI*randm11r(&se);     // " y_j
      z[j] = 1.0 + M_PI*randm11r(&se);      // " z_j
    }
  }
  FLT* s = (FLT*)malloc(sizeof(FLT)*N);    // targ freqs (1-cmpt)
  FLT* t = (FLT*)malloc(sizeof(FLT)*N);    // targ freqs (2-cmpt)
  FLT* u = (FLT*)malloc(sizeof(FLT)*N);    // targ freqs (3-cmpt)
  FLT S1 = (FLT)N1/2;                   // choose freq range sim to type 1
  FLT S2 = (FLT)N2/2;
  FLT S3 = (FLT)N3/2;
#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(dynamic,CHUNK)
    for (INT k=0; k<N; ++k) {
      s[k] = S1*(1.7 + randm11r(&se));  //S*(1.7 + k/(FLT)N); // offset the freqs
      t[k] = S2*(-0.5 + randm11r(&se));
      u[k] = S3*(0.9 + randm11r(&se));
    }
  }
  timer.restart();
  ier = finufft3d3(M,x,y,z,c,isign,tol,N,s,t,u,F,opts);
  ti=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
    exit(ier);
  } else
    printf("\t%ld NU to %ld NU in %.3g s   %.3g srcs/s, %.3g targs/s\n",(INT64)M,(INT64)N,ti,M/ti,N/ti);

  INT kt = N/2;          // check arbitrary choice of one targ pt
  Ft = (0,0);
  for (INT j=0;j<M;++j)
    Ft += c[j] * exp(ima*(FLT)isign*(s[kt]*x[j] + t[kt]*y[j] + u[kt]*z[j]));
  printf("one targ: rel err in F[%ld] is %.3g\n",(INT64)kt,abs(Ft-F[kt])/infnorm(N,F));
  if ((INT64)M*N<=BIGPROB) {                  // also full direct eval
    CPX* Ft = (CPX*)malloc(sizeof(CPX)*N);
    dirft3d3(M,x,y,z,c,isign,N,s,t,u,Ft);       // writes to F
    printf("dirft3d: rel l2-err of result F is %.3g\n",relerrtwonorm(N,Ft,F));
    //cout<<"s t u, F, Ft, F/Ft:\n"; for (int k=0;k<N;++k) cout<<s[k]<<" "<<t[k]<<" "<<u[k]<<", "<<F[k]<<",\t"<<Ft[k]<<",\t"<<F[k]/Ft[k]<<endl;
    free(Ft);
  }

  free(x); free(y); free(z); free(c); free(F); free(s); free(t); free(u);
  return ier;
}
