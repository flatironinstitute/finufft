#include "../src/finufft_guru.h"
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
  int optsDebug{0};
  int sprDebug{0};
  if (argc>6) sscanf(argv[6],"%d",&optsDebug);
  sprDebug = (optsDebug>1) ? 1 : 0;  // see output from spreader
  int sprSort{2};
  if (argc>7) sscanf(argv[7],"%d",&sprSort);

  if (argc>8) sscanf(argv[8],"%lf",&upsampfac);

  if (argc==1 || argc==2 || argc>9) {
    fprintf(stderr,"Usage: finufft_guru_test [N1 N2 [Nsrc [howMany [tol [debug [spread_sort [upsampfac]]]]]]\n");
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

  printf("test guru interface:\n"); // -------------- type 1

  finufft_plan plan;

  BIGINT n_modes[3] {N1, N2, 1}; //#modes per dimension 

  int n_dims = 2;
  CNTime timer; timer.start();
  int ier = make_finufft_plan(type1, n_dims,  n_modes, isign, howMany,tol, &plan);
  double plan_t = timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else{
      printf("finufft_plan creation for %lldx%lld modes completed in %.3g s\n", (long long)M, (long long)M, plan_t);
  }

  plan.opts.upsampfac=(FLT)upsampfac;
  plan.opts.debug = optsDebug;
  plan.opts.spread_debug = sprDebug;
  plan.spopts.debug = sprDebug;
  plan.opts.spread_sort = sprSort;
  plan.opts.upsampfac = upsampfac;
  
  timer.restart();
  ier = setNUpoints(&plan, M, x, y, NULL, NULL);
  double sort_t = timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else{
    printf("set NU points for %lldx%lld src points completed in %.3g s\n", (long long)N1, (long long)N2, sort_t);
  }
  timer.restart();
  ier = finufft_exec(&plan,c,F);
  double  exec_t=timer.elapsedsec();

  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("execute %d of: %lld NU pts to (%lld,%lld) modes in %.3g s or \t%.3g NU pts/s\n", howMany, 
	   (long long)M,(long long)N1,(long long)N2,exec_t,howMany*M/exec_t);

  //Error Checking 
  BIGINT nt1 = (BIGINT)(0.37*N1), nt2 = (BIGINT)(0.26*N2);  // choose some mode index to check
  CPX Ft = CPX(0,0), J = IMA*(FLT)isign;
  CPX Ft_last(0,0);

  for (BIGINT j=0; j<M; ++j){
    Ft += c[j] * exp(J*(nt1*x[j]+nt2*y[j]));   // crude direct
    if(howMany > 1){
      Ft_last += c[j+ (howMany-1)*M] *  exp(J*(nt1*x[j]+nt2*y[j]));
    }
  }
  BIGINT it = N1/2+nt1 + N1*(N2/2+nt2);   // index in complex F as 1d array
  printf("one mode in first trial: rel err in F[%lld,%lld] is %.3g\n",(long long)nt1,(long long)nt2,abs(Ft-F[it])/infnorm(N,F));

  if(howMany > 1)
    printf("one mode in last trial: rel err in F[%lld,%lld] is %.3g\n",(long long)nt1,(long long)nt2,abs(Ft_last-F[it+(howMany-1)*M])/infnorm(N,F));
  
  if ((int64_t)M*N<=BIGPROB) {                   // also check vs full direct eval
    CPX* Ft = (CPX*)malloc(sizeof(CPX)*N);
    if(Ft){ 
      dirft2d1(M,x,y,c,isign,N1,N2,Ft);
      printf("dirft2d: rel l2-err of result F is %.3g\n",relerrtwonorm(N,Ft,F));
      free(Ft);
    }
  }
 free(F);
 
  CPX* F_comp = (CPX*)malloc(sizeof(CPX)*N*howMany);   // mode ampls
  if(!F_comp)
    printf("failed to malloc comparison result array\n");

  else{

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
  //compare the result with finufft2dmany
  printf("test finufft2d1many interface\n");
  FFTW_FORGET_WISDOM();
  timer.restart();
  ier = finufft2d1many(howMany, M, x, y, c, isign , tol, N1, N2, F_comp, plan.opts);
  double t_comp =timer.elapsedsec();
  if(ier!=0){
    printf("error (ier=%d)!\n", ier);
  }
  else{
    printf("    %d of: %lld NU pts to (%lld,%lld) modes in %.3g s \t%.3g NU pts/s\n", howMany,(long long)M,(long long)N1,(long long)N2,t_comp,howMany*M/t_comp);
  
    printf("\tspeedup (T_finufft_guru/T_finufft2d1many) = %.3g\n", t_comp/(plan_t + sort_t + exec_t));
  }
  
  }
  free(x); free(y); free(c);
  finufft_destroy(&plan);
  return ier;
}
