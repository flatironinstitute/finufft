#include <finufft.h>
#include <finufft_old.h>
#include <dirft.h>
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


//forward declaration 
double runOldFinufft(CPX *c,CPX *F,finufft_plan *plan);
finufft_type intToType(int i);
int typeToInt(finufft_type type);

  
int main(int argc, char* argv[])
/* Test/Demo the guru interface

   Usage: finufftGuru_test [ntransf [type [ndim [Nmodes1 Nmodes2 Nmodes3 [Nsrc [tol [debug [spread_sort [upsampfac]]]]]]]

   debug = 0: rel errors and overall timing, 1: timing breakdowns
           2: also spreading output

   Example: finufftGuru_test 1 1 2 1000 1000 0 1000000 1e-12 2 2.0
*/
  
{
  BIGINT M = 1e6, N1 = 1000, N2 = 500, N3=250;  // defaults: M = # srcs, N1,N2 = # modes
  double w, tol = 1e-6;          // default
  double upsampfac = 2.0;        // default
  int ntransf = 1;
  int ndim = 1;
  finufft_type type = type1;
  int i;
  int isign = +1;             // choose which exponential sign to test


  /************************************************************************************************************/
  /* Collect command line arguments
  /************************************************************************************************************/

  if (argc>1) 
    sscanf(argv[1],"%d",&i); ntransf = i;
  if(argc > 2)
    sscanf(argv[2],"%d",&i); type = intToType(i);

  if(argc > 3)
    sscanf(argv[3],"%d",&i); ndim = i;    

  if(argc > 4){
    sscanf(argv[4],"%lf",&w); N1 = (BIGINT)w;
    sscanf(argv[5],"%lf",&w); N2 = (BIGINT)w;
    sscanf(argv[6],"%lf",&w); N3 = (BIGINT)w;
  }
  if (argc>7) { sscanf(argv[7],"%lf",&w); M = (BIGINT)w; }
  if (argc>8) {
    sscanf(argv[8],"%lf",&tol);
    if (tol<=0.0) { printf("tol must be positive!\n"); return 1; }
  }
  int optsDebug{0};
  int sprDebug{0};
  if (argc>9) sscanf(argv[9],"%d",&optsDebug);
  sprDebug = (optsDebug>1) ? 1 : 0;  // see output from spreader
  int sprSort{2};
  if (argc>10) sscanf(argv[10],"%d",&sprSort);

  if (argc>11) sscanf(argv[11],"%lf",&upsampfac);

  if (argc==1 || argc>12) {
    fprintf(stderr,"Usage: finufftGuru_test [ntransf [type [ndim [N1 N2 N3 [Nsrc [tol [debug [spread_sort [upsampfac]]]]]]\n");
    return 1;
  }

  /************************************************************************************************************/
  /*  Allocate and initialize input:
  /************************************************************************************************************/
  
  cout << scientific << setprecision(15);
  N2 = (N2 == 0) ? 1 : N2;
  N3 = (N3 == 0) ? 1 : N3;
  
  BIGINT N = N1*N2*N3;
  
  FLT *x = (FLT *)malloc(sizeof(FLT)*M);        // NU pts x coords
  if(!x){
    fprintf(stderr, "failed malloc x coords");
    return 1;
  }

  FLT *y = NULL;
  FLT *z = NULL;
  if(ndim > 1){
    y = (FLT *)malloc(sizeof(FLT)*M);        // NU pts y coords
    if(!y){
      fprintf(stderr, "failed malloc y coords");
      free(x);
      return 1;
    }
  }

  if(ndim > 2){
    z = (FLT *)malloc(sizeof(FLT)*M);        // NU pts z coords
    if(!z){
      fprintf(stderr, "failed malloc z coords");
      free(x);
      if(y)
	free(y);
      return 1;
    }
  }

  
  FLT* s = NULL; 
  FLT* t = NULL; 
  FLT* u = NULL;

  if(type == type3){
    s = (FLT*)malloc(sizeof(FLT)*N);    // targ freqs (1-cmpt)
    FLT S1 = (FLT)N1/2;            

#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();  // needed for parallel random #s
#pragma omp for schedule(dynamic,CHUNK)
    for (BIGINT k=0; k<N; ++k) {
      s[k] = S1*(1.7 + randm11r(&se));  
    }
    
    if(ndim > 1 ){
      t = (FLT*)malloc(sizeof(FLT)*N);    // targ freqs (2-cmpt)
      FLT S2 = (FLT)N2/2;
      
#pragma omp for schedule(dynamic,CHUNK)
      for (BIGINT k=0; k<N; ++k) {
	t[k] = S2*(1.7 + randm11r(&se));  
      }
    }
    
    if(ndim > 2){
      u = (FLT*)malloc(sizeof(FLT)*N);    // targ freqs (3-cmpt)
      FLT S3 = (FLT) N3/2;

#pragma omp for schedule(dynamic,CHUNK)
      for (BIGINT k=0; k<N; ++k) {
	u[k] = S3*(1.7 + randm11r(&se));  
      }
    }
  }
  }      
  
  CPX* c = (CPX*)malloc(sizeof(CPX)*M*ntransf);   // strengths 
  if(!c){
    fprintf(stderr, "failed malloc strengths");
    free(x);
    if(y)
      free(y);
    if(z)
      free(z);
    return 1;
  }

  CPX* F = (CPX*)malloc(sizeof(CPX)*N*ntransf);   // mode ampls
  if(!F){
    fprintf(stderr, "failed malloc result array!");
    free(x);
    if(y)
      free(y);
    if(z)
      free(z);
    free(c); 
    return 1;
  }
  
#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();  // needed for parallel random #s
#pragma omp for schedule(dynamic,CHUNK)
    for (BIGINT j=0; j<M; ++j) {
      x[j] = M_PI*randm11r(&se);
      if(y)
	y[j] = M_PI*randm11r(&se);
      if(z)
	z[j] = M_PI*randm11r(&se);
    }
#pragma omp for schedule(dynamic,CHUNK)
    for(BIGINT i = 0; i<ntransf*M; i++)
	c[i] = crandm11r(&se);
  }

  //clean slate
  FFTW_FORGET_WISDOM();
    
  /**********************************************************************************************/
  /* Finufft
  /**********************************************************************************************/

  //Start by instantiating a finufft_plan
  finufft_plan plan;

  //Guru Step 0
  finufft_default_opts(&plan.opts);
  
  //Optional Customization opts 
  plan.opts.upsampfac=(FLT)upsampfac;
  plan.opts.debug = optsDebug;
  plan.opts.spread_debug = sprDebug;
  plan.spopts.debug = sprDebug;
  plan.opts.spread_sort = sprSort;
  plan.opts.upsampfac = upsampfac;

  
  BIGINT n_modes[3] {N1, N2, N3}; //#modes per dimension 

  CNTime timer; timer.start();

  //Guru Step 1
  int ier = make_finufft_plan(type, ndim,  n_modes, isign, ntransf,tol, &plan);
  double plan_t = timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else{
    printf("finufft_plan creation for %lld modes completed in %.3g s\n", (long long)N, plan_t);
  }

  
  timer.restart();
  //Guru Step 2
  ier = setNUpoints(&plan, M, x, y, z, s, t, u);
  double sort_t = timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else{
    printf("set NU points for %lld src points completed in %.3g s\n", (long long)M, sort_t);
  }
  
  timer.restart();
  //Guru Step 3
  ier = finufft_exec(&plan,c,F);
  double  exec_t=timer.elapsedsec();

  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("execute %d of: %lld NU pts to %lld modes in %.3g s or \t%.3g NU pts/s\n", ntransf, 
	   (long long)M,(long long)N, exec_t , ntransf*M/exec_t);

  //Guru Step 4
  //finufft_destroy(&plan);

  //Don't forget to destroy, but in this instance we'll wait and save the plan
  //for the timing comparison step

  //You're done!
  
  /**********************************************************************************************/
  /* Timing Comparisons 
  /**********************************************************************************************/

  double totalTime = plan_t + sort_t + exec_t;
  //comparing timing results with repeated calls to corresponding finufft function 
  FFTW_FORGET_WISDOM();

  double oldTime = runOldFinufft(c,F, &plan);
  FFTW_FORGET_WISDOM();

  printf("\tspeedup (T_finufft[%d]d[%d]_old / T_finufft[%d]d[%d]) = %.3g\n", typeToInt(type), ndim,
	 typeToInt(type), ndim, oldTime/totalTime);
  
  //???
  //runDirectComputation(c,F,plan);
  //FFTW_FORGET_WISDOM();
  
  /**********************************************************************************************/
  /* Free Memory
  /*******************************************************************************************/

  finufft_destroy(&plan);

  free(F);
  free(c);
  free(x); 
  if(y)
    free(y);
  if(z)
    free(z);

  if(s)
    free(s);
  if(t)
    free(t);
  if(u)
    free(u);
}





  
