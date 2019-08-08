#include <finufft_tempinstant.h>
#include <finufft_old.h>
#include <helpers.h>
#include <fftw_defs.h>
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
double runOldFinufft(TEMPLATE(CPX, float) *c,TEMPLATE(CPX, float) *F,TEMPLATE(finufft_plan, float) *plan);
finufft_type intToType(int i);
int typeToInt(finufft_type type);

  
int main(int argc, char* argv[])
/* Test/Demo the guru interface

   Usage: finufftGuru_test [ntransf [type [ndim [Nmodes1 Nmodes2 Nmodes3 [Nsrc [tol [debug [do_spread [upsampfac]]]]]]]

   debug = 0: rel errors and overall timing, 1: timing breakdowns
           2: also spreading output

   Example: finufftGuru_test 1 1 2 1000 1000 0 1000000 1e-12 2 1 2.0

   For Type3, please enter nk in Nmodes space, 0 for rest
   Example w/ nk = 5000: finufftGuru_test 1 3 2 5000 0 0 1000000 1e-12 2 1 2.0
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
  int optsDebug = 0;
  int sprDebug = 0;
  if (argc>9) sscanf(argv[9],"%d",&optsDebug);
  sprDebug = (optsDebug>1) ? 1 : 0;  // see output from spreader
  int sprSort = 2 ;
  if (argc>10) sscanf(argv[10],"%d",&sprSort);

  if (argc>11) sscanf(argv[11],"%lf",&upsampfac);

  if (argc==1 || argc>12) {
    fprintf(stderr,"Usage: finufftGuru_test [ntransf [type [ndim [N1 N2 N3 [Nsrc [tol [debug [do_spread [upsampfac]]]]]]\n");
    return 1;
  }

  /************************************************************************************************************/
  /*  Allocate and initialize input:
  /************************************************************************************************************/
  
  cout << scientific << setprecision(15);
  N2 = (N2 == 0) ? 1 : N2;
  N3 = (N3 == 0) ? 1 : N3;
  
  BIGINT N = N1*N2*N3;
  
  float *x = (float *)malloc(sizeof(float)*M);        // NU pts x coords
  if(!x){
    fprintf(stderr, "failed malloc x coords\n");
    return 1;
  }

  float *y = NULL;
  float *z = NULL;
  if(ndim > 1){
    y = (float *)malloc(sizeof(float)*M);        // NU pts y coords
    if(!y){
      fprintf(stderr, "failed malloc y coords\n");
      free(x);
      return 1;
    }
  }

  if(ndim > 2){
    z = (float *)malloc(sizeof(float)*M);        // NU pts z coords
    if(!z){
      fprintf(stderr, "failed malloc z coords\n");
      free(x);
      if(y)
	free(y);
      return 1;
    }
  }

  
  float* s = NULL; 
  float* t = NULL; 
  float* u = NULL;

  if(type == type3){
    s = (float*)malloc(sizeof(float)*N);    // targ freqs (1-cmpt)
    float S1 = (float)N1/2;            

#pragma omp parallel
  {
    unsigned int se=MY_OMP_GET_THREAD_NUM();  // needed for parallel random #s
#pragma omp for schedule(dynamic,CHUNK)
    for (BIGINT k=0; k<N; ++k) {
      s[k] = S1*(1.7 + randm11r(&se));  
    }
    
    if(ndim > 1 ){
      t = (float*)malloc(sizeof(float)*N);    // targ freqs (2-cmpt)
      float S2 = (float)N2/2;
      
#pragma omp for schedule(dynamic,CHUNK)
      for (BIGINT k=0; k<N; ++k) {
	t[k] = S2*(1.7 + randm11r(&se));  
      }
    }
    
    if(ndim > 2){
      u = (float*)malloc(sizeof(float)*N);    // targ freqs (3-cmpt)
      float S3 = (float) N3/2;

#pragma omp for schedule(dynamic,CHUNK)
      for (BIGINT k=0; k<N; ++k) {
	u[k] = S3*(1.7 + randm11r(&se));  
      }
    }
  }
  }      
  
  TEMPLATE(CPX, float)* c = (TEMPLATE(CPX, float)*)malloc(sizeof(TEMPLATE(CPX, float))*M*ntransf);   // strengths 
  if(!c){
    fprintf(stderr, "failed malloc strengths array allocation \n");
    free(x);
    if(y)
      free(y);
    if(z)
      free(z);
    return 1;
    if(s)
      free(s);
    if(t)
      free(t);
    if(u)
      free(u);
  }

  TEMPLATE(CPX, float)* F = (TEMPLATE(CPX, float)*)malloc(sizeof(TEMPLATE(CPX, float))*N*ntransf);   // mode ampls
  if(!F){
    fprintf(stderr, "failed malloc result array!\n");
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
  TEMPLATE(FFTW_FORGET_WISDOM, float)();
    
  /**********************************************************************************************/
  /* Finufft
  /**********************************************************************************************/

  printf("------------------------GURU INTERFACE------------------------------\n");
  //Start by instantiating a finufft_plan
  TEMPLATE(finufft_plan, float) plan;

  //Guru Step 0
  finufft_default_opts(&plan.opts);
  
  //Optional Customization opts 
  plan.opts.upsampfac=(float)upsampfac;
  plan.opts.debug = optsDebug;
  plan.opts.spread_debug = sprDebug;
  plan.spopts.debug = sprDebug;
  plan.opts.spread_sort = sprSort;
  plan.opts.upsampfac = upsampfac;

  BIGINT n_modes[3];
  n_modes[0] = N1;
  n_modes[1] = N2;
  n_modes[2] = N3; //#modes per dimension 

  CNTime timer; timer.start();

  int blksize = MY_OMP_GET_MAX_THREADS(); 
  
  //Guru Step 1
  int ier = TEMPLATE(make_finufft_plan,float)(type, ndim,  n_modes, isign, ntransf, tol, blksize, &plan);
  //for type3, omit n_modes and send in NULL
  
  double plan_t = timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
    return ier;
  } else{
    printf("finufft_plan creation for %lld modes completed in %.3g s\n", (long long)N, plan_t);
  }

  
  timer.restart();
  //Guru Step 2
  ier = TEMPLATE(setNUpoints,float)(&plan, M, x, y, z, N, s, t, u); //type 1+2, N=0, s,t,u = NULL
  double sort_t = timer.elapsedsec();
  if (ier) {
    printf("error (ier=%d)!\n",ier);
    return ier;
  } else{
    printf("set NU points for %lld src points completed in %.3g s\n", (long long)M, sort_t);
  }
  
  timer.restart();
  //Guru Step 3
  ier = TEMPLATE(finufft_exec,float)(&plan,c,F);
  double  exec_t=timer.elapsedsec();

  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
    return ier;
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
  TEMPLATE(FFTW_FORGET_WISDOM, float)();

 printf("------------------------OLD IMPLEMENTATION------------------------------\n");
  
  double oldTime = runOldFinufft(c,F, &plan);
  TEMPLATE(FFTW_FORGET_WISDOM, float)();

  printf("execute %d of: %lld NU pts to %lld modes in %.3g s or \t%.3g NU pts/s\n", ntransf, 
	   (long long)M,(long long)N, oldTime , ntransf*M/oldTime);
  
  printf("\tspeedup (T_finufft[%d]d[%d]_old / T_finufft[%d]d[%d]) = %.3g\n", ndim,  typeToInt(type),
	  ndim,typeToInt(type), oldTime/totalTime);
  
  
  /**********************************************************************************************/
  /* Free Memory
  /*******************************************************************************************/

  TEMPLATE(finufft_destroy,float)(&plan);

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





  
