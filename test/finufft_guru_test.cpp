#include "../src/finufft_guru.h"
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
/* Test executable for finufft in 2d, using the guru interface

   Usage: finufft2d_test [Nmodes1 Nmodes2 [Nsrc [howMany [tol [debug [spread_sort [upsampfac]]]]]]]

   debug = 0: rel errors and overall timing, 1: timing breakdowns
           2: also spreading output

   Example: finufft2d_guru_test 1000 1000 1000000 1e-12 1 2 2.0

   Malleo 6/7/19
*/
{
  BIGINT M = 1e6, N1 = 1000, N2 = 500;  // defaults: M = # srcs, N1,N2 = # modes
  double w, tol = 1e-6;          // default
  double upsampfac = 2.0;        // default
  int howMany {1};
  nufft_opts opts; finufft_default_opts(&opts);
  opts.debug = 0;            // 1 to see some timings
  int isign = +1;             // choose which exponential sign to test
  if (argc>1) {
    sscanf(argv[1],"%lf",&w); N1 = (BIGINT)w;
    sscanf(argv[2],"%lf",&w); N2 = (BIGINT)w;
  }
  if (argc>3) { sscanf(argv[3],"%lf",&w); M = (BIGINT)w; }
  if (argc>4) { sscanf(argv[4], "%d", &howMany); }
  if (argc>5) {
    sscanf(argv[5],"%lf",&tol);
    if (tol<=0.0) { printf("tol must be positive!\n"); return 1; }
  }
  if (argc>6) sscanf(argv[6],"%d",&opts.debug);
  opts.spread_debug = (opts.debug>1) ? 1 : 0;  // see output from spreader
  if (argc>7) sscanf(argv[7],"%d",&opts.spread_sort);
  if (argc>8) sscanf(argv[8],"%lf",&upsampfac);
  opts.upsampfac=(FLT)upsampfac;
  if (argc==1 || argc==2 || argc>9) {
    fprintf(stderr,"Usage: finufft2d_test [N1 N2 [Nsrc [howMany [tol [debug [spread_sort [upsampfac]]]]]]\n");
    return 1;
  }

  cout << scientific << setprecision(15);
  BIGINT N = N1*N2;
  
  FLT *x = (FLT *)malloc(sizeof(FLT)*M);        // NU pts x coords
  if(!x){
    fprintf(stderr, "failed malloc x coords");
    return 1;
  }

  FLT *y = (FLT *)malloc(sizeof(FLT)*M);        // NU pts y coords
  if(!y){
    fprintf(stderr, "failed malloc y coords");
    free(x);
    return 1;
  }

  CPX* c = (CPX*)malloc(sizeof(CPX)*M*howMany);   // strengths 
  if(!c){
    fprintf(stderr, "failed malloc strengths");
    free(x);
    free(y);
    return 1;
  }

  CPX* F = (CPX*)malloc(sizeof(CPX)*N*howMany);   // mode ampls
  if(!F){
    fprintf(stderr, "failed malloc result array!");
    free(x);
    free(y);
    free(c); 
    return 1;
  }
  
#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();  // needed for parallel random #s
#pragma omp for schedule(dynamic,CHUNK)
    for (BIGINT j=0; j<M; ++j) {
      x[j] = M_PI*randm11r(&se);
      y[j] = M_PI*randm11r(&se);
    }
#pragma omp for schedule(dynamic,CHUNK)
    for(BIGINT i = 0; i<howMany*M; i++)
	c[i] = crandm11r(&se);
  }

  printf("test 2d type-1:\n"); // -------------- type 1

  finufft_plan plan;

  BIGINT n_modes[3] {N1, N2, 1}; //#modes per dimension 
  BIGINT n_srcpts[3] {M,M,1}; //# pts per dimension

  CNTime timer; timer.start();
  int ier = make_finufft_plan(finufft_type::type1, 2, &n_srcpts[0], &n_modes[0], isign, howMany,tol, plan);
  double t1 = timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("finufft_plan creation for %lldx%lld modes \n completed in %.3g s\n", (long long)M, (long long)M, t1);
  

  timer.restart();
  ier = sortNUpoints(plan, x, y, NULL, NULL);
  t1 = timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("set NU points for %lldx%lld src points completed in %.3g s\n", (long long)N1, (long long)N2, t1);

  timer.restart();
  ier = finufft_exec(plan,c,F);
  t1=timer.elapsedsec();

  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("%lld NU pts to (%lld,%lld) modes in %.3g s or \t%.3g NU pts/s\n",
	   (long long)M,(long long)N1,(long long)N2,t1,M/t1);


  //Error Checking 
  BIGINT nt1 = (BIGINT)(0.37*N1), nt2 = (BIGINT)(0.26*N2);  // choose some mode index to check
  CPX Ft = CPX(0,0), J = IMA*(FLT)isign;
  for (BIGINT j=0; j<M; ++j)
    Ft += c[j] * exp(J*(nt1*x[j]+nt2*y[j]));   // crude direct
  BIGINT it = N1/2+nt1 + N1*(N2/2+nt2);   // index in complex F as 1d array
  printf("one mode: rel err in F[%lld,%lld] is %.3g\n",(long long)nt1,(long long)nt2,abs(Ft-F[it])/infnorm(N,F));
  if ((int64_t)M*N<=BIGPROB) {                   // also check vs full direct eval
    CPX* Ft = (CPX*)malloc(sizeof(CPX)*N);
    if(Ft){ 
      dirft2d1(M,x,y,c,isign,N1,N2,Ft);
      printf("dirft2d: rel l2-err of result F is %.3g\n",relerrtwonorm(N,Ft,F));
      free(Ft);
    }
  }
  free(x); free(y); free(c); free(F);
  finufft_destroy(plan);
  return ier;
}
