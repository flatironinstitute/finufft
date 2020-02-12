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
/* Test executable for finufft in 1d many interface, types 1,2, and 3.

   Usage: finufft2dmany_test [ntransf [Nmodes [Nsrc [tol [debug [spreadsort [upsampfac]]]]]]]

   debug = 0: rel errors and overall timing, 1: timing breakdowns
           2: also spreading output

   Example: finufft1dmany_test 1000 1e2 1e4 1e-6 1 2 2.0
*/
 {

   BIGINT M = 1e6, N = 1000;  // defaults: M = # srcs, N = # modes
  int debug = 0;
  int ntransf = 400;                      // # of vectors for "many" interface

  double w, tol = 1e-6;          // default
  double upsampfac = 2.0;        // default
  nufft_opts opts; finufft_default_opts(&opts);
  opts.debug = 0;            // 1 to see some timings
  // opts.fftw = FFTW_MEASURE;  // change from usual FFTW_ESTIMATE
  int isign = +1;             // choose which exponential sign to test
  if (argc>1) { sscanf(argv[1],"%lf",&w); ntransf = (int)w; }
  if (argc>2) {
    sscanf(argv[2],"%lf",&w); N = (BIGINT)w;
  }
  if (argc>3) { sscanf(argv[3],"%lf",&w); M = (BIGINT)w; }
  if (argc>4) {
    sscanf(argv[4],"%lf",&tol);
    if (tol<=0.0) { printf("tol must be positive!\n"); return 1; }
  }
  if (argc>5) sscanf(argv[5],"%d",&debug);
  opts.debug = debug;
  opts.spread_debug = (debug>1) ? 1 : 0;  // see output from spreader
  if (argc>6) sscanf(argv[6],"%d",&opts.spread_sort);
  if (argc>7) sscanf(argv[7],"%lf",&upsampfac);
  opts.upsampfac=(FLT)upsampfac;

  if (argc==1 || argc==2 || argc>8) {
    fprintf(stderr,"Usage: finufft1d_test [ntransf [Nmodes [Nsrc [tol [debug [upsampfac]]]]]]\n");
    return 1;
  }
  cout << scientific << setprecision(15);
 
  FLT* x = (FLT*)malloc(sizeof(FLT)*M);  // NU pts x coords
  CPX* c = (CPX*)malloc(sizeof(CPX)*M*ntransf);   // strengths 
  CPX* F = (CPX*)malloc(sizeof(CPX)*N*ntransf);   // mode ampls

#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(dynamic,CHUNK)
    for (BIGINT j=0; j<M; ++j) {
      x[j] = M_PI*randm11r(&se);
    }
#pragma omp for schedule(dynamic,CHUNK)
    for (BIGINT j = 0; j<ntransf*M; ++j)
    {
        c[j] = crandm11r(&se);
    }
  }


  printf("------------------test 1dmany type-1:------------------\n"); // -------------- type 1
  CNTime timer; timer.start();
  int ier = finufft1d1many(ntransf,M,x,c,isign,tol,N,F,&opts);
  double ti=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("    %d of: %lld NU pts to %lld modes in %.3g s or  \t%.3g NU pts/s\n", ntransf,(long long)M,(long long)N,ti,ntransf*M/ti);

  
  
  int d = (ntransf-1);    // choose a trial to check
  BIGINT nt1 = (BIGINT)(0.37*N);  // choose some mode index to check
  CPX Ft = CPX(0,0), J = IMA*(FLT)isign;
  for (BIGINT j=0; j<M; ++j)
    Ft += c[j+d*M] * exp(J*(nt1*x[j]));   // crude direct
  BIGINT it = N/2+nt1 ; // index in complex F as 1d array
  printf("[err check] one mode: rel err in F[%lld] of data[%d] is %.3g\n",
	 (long long)nt1,d,abs(Ft-F[it+d*N])/infnorm(N,F+d*N));

  //check against the old
  CPX * F_old = (CPX *)malloc(sizeof(CPX)*N*ntransf);
  CPX * F_start;
  CPX * c_start;
  timer.restart();
  opts.debug = 0;
  opts.spread_debug = 0; 
  for(BIGINT j = 0; j < ntransf; j++){
    F_start = F_old + j*N;
    c_start = c + j*M;
    finufft1d1_old(M,x,c_start,isign,tol,N,F_start,opts);
  }
  double t = timer.elapsedsec();
  printf("[speedup] \t (T_finufft1d1 / T_finufft1d1many) = %.3g\n", t/ti);
  
  printf("[err check] finufft1d1_old: rel l2-err of result F is %.3g\n",relerrtwonorm(N,F_old,F));
  printf("[err check] on trial %d one mode: rel err in F[%lld] is %.3g\n",d,(long long)nt1,abs(F_old[N/2+nt1 + d*N]-F[N/2+nt1+d*N])/infnorm(N,F+d*N));
  free(F_old);
 


  printf("------------------test 1dmany type-2:------------------\n"); // -------------- type 2

  opts.debug = debug;
  opts.spread_debug = (debug>1) ? 1 : 0;  // see output from spreader

#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();  // needed for parallel random #s
#pragma omp for schedule(dynamic,CHUNK)
    for (BIGINT m=0; m<N; ++m) F[m] = crandm11r(&se);
  }
  timer.restart();
  ier = finufft1d2many(ntransf, M,x,c,isign,tol,N,F,&opts);
  //cout<<"c:\n"; for (int j=0;j<M;++j) cout<<c[j]<<endl;
  ti=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
    exit(ier);
  } else
    printf("\t%lld modes to %lld NU pts in %.3g s \t%.3g NU pts/s\n",(long long)N,(long long)M,ti,M/ti);
 

  BIGINT jt = M/2;          // check arbitrary choice of one targ pt
  CPX ct = CPX(0,0);
  BIGINT m=0, k0 = N/2;          // index shift in fk's = mag of most neg freq
  //#pragma omp parallel for schedule(dynamic,CHUNK) reduction(cmplxadd:ct)
  for (BIGINT m1=-k0; m1<=(N-1)/2; ++m1)
    ct += F[d*N + m++] * exp(IMA*((FLT)(isign*m1))*x[jt]);   // crude direct
  printf("[err check] one targ: rel err in c[%lld] is %.3g\n",(long long)jt,abs(ct-c[jt + d*M])/infnorm(M,c+d*M));

  opts.debug = 0;
  opts.spread_debug = 0;
  //check against the old
  CPX * c_old = (CPX *)malloc(sizeof(CPX)*M*ntransf);
  timer.restart();
  for(BIGINT j = 0; j < ntransf; j++){
    F_start = F + j*N;
    c_start = c_old + j*M;
    finufft1d2_old(M,x,c_start,isign,tol,N,F_start,opts);
  }
  t = timer.elapsedsec();
  printf("[speedup] \t (T_finufft1d2 / T_finufft1d2many) = %.3g\n", t/ti);

  printf("[err check] finufft1d2_old: rel l2-err of result c is %.3g\n",relerrtwonorm(M,c_old+d*M,c+d*M));
  printf("[err check] on trial %d one targ: rel err in c[%lld] is %.3g\n",d, (long long)jt,abs(c_old[jt+d*M]-c[jt+d*M])/infnorm(M,c+d*M));
  free(c_old);

  printf("------------------test 1dmany type-3:------------------\n"); // -------------- type 3

  opts.debug = debug;
  opts.spread_debug = (debug>1) ? 1 : 0;  // see output from spreader

#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(dynamic,CHUNK)
    for (BIGINT j=0; j<M; ++j) x[j] = 2.0 + PI*randm11r(&se);  // new x_j srcs
  }
  FLT* s = (FLT*)malloc(sizeof(FLT)*N);    // targ freqs
  FLT S = (FLT)N/2;                   // choose freq range sim to type 1
#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();
#pragma omp for schedule(dynamic,CHUNK)
    for (BIGINT k=0; k<N; ++k) s[k] = S*(1.7 + randm11r(&se)); //S*(1.7 + k/(FLT)N); // offset
  
#pragma omp for schedule(dynamic,CHUNK)
    for (BIGINT j = 0; j<ntransf*M; ++j)
    {
        c[j] = crandm11r(&se);
    }

  }
  
  timer.restart();
  ier = finufft1d3many(ntransf, M,x,c,isign,tol,N,s,F,&opts);
  ti=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
    exit(ier);
  } else
    printf("\t%lld NU to %lld NU in %.3g s   %.3g srcs/s, %.3g targs/s\n",(long long)M,(long long)N,t,M/ti,N/ti);

  
  BIGINT kt = N/4;          // check arbitrary choice of one targ pt
  Ft = CPX(0,0);
  //#pragma omp parallel for schedule(dynamic,CHUNK) reduction(cmplxadd:Ft)
  for (BIGINT j=0;j<M;++j)
    Ft += c[j+d*M] * exp(IMA*(FLT)isign*s[kt]*x[j]);
  printf("[err check] one targ: rel err against direct in F[%lld] is %.3g\n",(long long)kt,abs(Ft-F[kt+d*N])/infnorm(N,F+d*N));


  opts.debug = 0;
  opts.spread_debug = 0;
  //check against the old
  CPX *F3_old = (CPX *)malloc(sizeof(CPX)*N*ntransf);
  timer.restart();
  for(int k = 0; k < ntransf; k++){
    c_start = c + k*M;
    F_start = F3_old + k*N;
    ier = finufft1d3_old(M,x,c_start,isign,tol,N,s,F_start, opts);
  }
  t = timer.elapsedsec();
  printf("[speedup] \t T_finufft1d2 / T_finufft1d2many) = %.3g\n", t/ti);

  
  printf("[err check] finufft1d3_old: rel l2-err of result c is %.3g\n",relerrtwonorm(N,F3_old+d*N,F+d*N));
  printf("[err check] one targ: rel err against old in F[%lld] is %.3g\n",(long long)kt,
	 abs(F3_old[kt+d*N]-F[kt+d*N])/infnorm(N,F+d*N));

  free(F3_old);
  free(x);
  free(s);
  free(c);
  free(F);
}  
