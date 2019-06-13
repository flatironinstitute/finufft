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

   Usage: finufftGuru2_test [Nmodes1 Nmodes2 [Nsrc [nvecs [tol [debug [spread_sort [upsampfac]]]]]]]

   debug = 0: rel errors and overall timing, 1: timing breakdowns
           2: also spreading output

   Example: finufftGuru2_test 1000 1000 1000000 1 1e-6 2 2.0

*/
{
  BIGINT M = 1e6, N1 = 1000, N2 = 500;  // defaults: M = # srcs, N1,N2 = # modes
  double w, tol = 1e-6;          // default
  double upsampfac = 2.0;        // default
  int nvecs {1};
  
  
  int isign = +1;             // choose which exponential sign to test
  if (argc>1) {
    sscanf(argv[1],"%lf",&w); N1 = (BIGINT)w;
    sscanf(argv[2],"%lf",&w); N2 = (BIGINT)w;
  }
  if (argc>3) { sscanf(argv[3],"%lf",&w); M = (BIGINT)w; }
  if (argc>4) { sscanf(argv[4], "%d", &nvecs); }
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
    fprintf(stderr,"Usage: finufftGuru2_test [N1 N2 [Nsrc [nvecs [tol [debug [spread_sort [upsampfac]]]]]]\n");
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

  CPX* c = (CPX*)malloc(sizeof(CPX)*M*nvecs);   // strengths 
  if(!c){
    fprintf(stderr, "failed malloc strengths");
    free(x);
    free(y);
    return 1;
  }

  CPX* F = (CPX*)malloc(sizeof(CPX)*N*nvecs);   // mode ampls
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
    for(BIGINT i = 0; i<nvecs*N; i++)
	F[i] = crandm11r(&se);
  }

  printf("test guru interface type 2:\n"); // -------------- type 2

  finufft_plan plan;

  BIGINT n_modes[3] {N1, N2, 1}; //#modes per dimension 

  int n_dims = 2;
  CNTime timer; timer.start();
  int ier = make_finufft_plan(type2, n_dims,  n_modes, isign, nvecs,tol, &plan);
  double plan_t = timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
    return -1;
  } else{
    printf("finufft_plan creation for %lld modes completed in %.3g s\n", (long long)N, plan_t);
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
    printf("execute %d of: %lld NU pts to (%lld,%lld) modes in %.3g s or \t%.3g NU pts/s\n", nvecs, 
	   (long long)M,(long long)N1,(long long)N2,exec_t,nvecs*M/exec_t);

  //Error Checking 
  BIGINT jt = M/2;  // choose some arbitrary target coordinate
  CPX ct(0,0), J = IMA*(FLT)isign;
  int d = floor(nvecs/2);  //testing first, and, if nvec >1,  middle round
  CPX ct_other = CPX(0,0);
  BIGINT m = 0;
  for (BIGINT m2=-(N2/2); m2<=(N2-1)/2; ++m2){
    for(BIGINT m1=-(N1/2); m1<=(N1-1)/2; ++m1){
      ct += F[m] * exp(J*(m1*x[jt]+m2*y[jt]));   // crude direct
      if(nvecs > 1){
	ct_other += F[m + d*N] *  exp(J*(m1*x[jt]+m2*y[jt]));
    }
      m++;
    }
  }
  printf("in nvec[%d]: rel err in c[%lld] is %.3g\n",0, (long long)jt, abs(ct-c[jt])/infnorm(M,c));

  if(nvecs > 1)
    printf("in nvec[%d]: rel err in c[%lld] is %.3g\n", d, (long long)jt,
	   abs(ct_other-c[jt+d*M])/infnorm(M, c+d*M));
  
  if ((int64_t)M*N<=BIGPROB) {                   // also check vs full direct eval
    CPX* ct_dir = (CPX*)malloc(sizeof(CPX)*M);
    if(ct_dir){ 
      dirft2d2(M,x,y,ct_dir,isign,N1,N2,F);
      printf("dirft2d: rel l2-err of result F is %.3g\n",relerrtwonorm(M,ct_dir,c));
      free(ct_dir);
    }
  }


 //compare timing results with finufft2dmany

 FFTW_FORGET_WISDOM();
 FLT maxerror{0.0};
 CPX* c_compMany = (CPX*)malloc(sizeof(CPX)*M*nvecs);   // resultant weights

 if(!c_compMany)
    printf("failed to malloc comparison result array\n");

  else{

    printf("test finufft2d2many interface\n");
    timer.restart();
    ier = finufft2d2many(nvecs, M, x, y, c_compMany, isign , tol, N1, N2, F, plan.opts);
    double t_compMany =timer.elapsedsec();

    if(ier!=0){
      printf("error (ier=%d)!\n", ier);
    }

    else{
      printf("    %d of:  (%lld,%lld) modes to %lld NU pts in %.3g s \t%.3g NU pts/s\n", nvecs,(long long)N1,(long long)N2,(long long)M,t_compMany,nvecs*M/t_compMany);
  
      printf("\tspeedup (T_finufft2d1many/T_finufft_guru) = %.3g\n", t_compMany/(plan_t + sort_t + exec_t));
    }

    //check accuracy (worst over the nvecs)
    for(int k = 0; k < nvecs; k++)
      maxerror = max(maxerror, relerrtwonorm(M, c_compMany + k*M, c + k*M));
    printf("err check vs many: sup( ||F_guru-F_many|| / ||F_many||_2 ) = %.3g\n", maxerror);
    
    
    free(c_compMany);
  }
  
  //comparing timing results with repeated finufft2d2
  FFTW_FORGET_WISDOM();
  CPX *cStart;
  CPX *fStart;

  CPX  *c_compSingle = (CPX *)malloc(sizeof(CPX)*M*nvecs);
  if(!c_compSingle){
    printf("failed to malloc result array for single finufft comparison\n");
  }
  else{
    //dial down output
    plan.opts.debug = 0;
    plan.opts.spread_debug = 0;
    timer.restart();
    for(int k = 0; k < nvecs; k++){
      cStart = c_compSingle + M*k;
      fStart = F + N*k;
      ier = finufft2d2(M, x, y, cStart, isign, tol, N1, N2, fStart, plan.opts);
    }
    
    double t_compSingle = timer.elapsedsec();
    printf("\tspeedup (T_finufft2d1/T_finufft_guru) = %.3g\n", t_compSingle/(plan_t + sort_t + exec_t));

    //check accuracy (worst over the nvecs)
    double max_error = 0.0;
    for(int k = 0; k < nvecs; k++)
      maxerror = max(maxerror, relerrtwonorm(M, c_compSingle + k*M, c + k*M));
    printf("err check vs non-many: sup( ||F_guru-F_single|| / ||F_single||_2 ) = %.3g\n", maxerror);
    
    free(c_compSingle);
    FFTW_FORGET_WISDOM();
  }
  free(F);  free(x); free(y); free(c);
  finufft_destroy(&plan);
  return ier;
}
