#include <finufft.h>
#include <defs.h>
#include <utils.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
// for sleep call
#include <unistd.h>

// how big a problem to do full direct DFT check in 3D...
#define BIGPROB 1e8

// for omp rand filling
#define CHUNK 1000000

//forward declarations for helper to (repeatedly if needed) call finufft?d?
double many_simple_calls(CPX *c,CPX *F,finufft_plan *plan);


// --------------------------------------------------------------------------
int main(int argc, char* argv[])
/* Test/Demo the guru interface

   Usage: finufftGuru_test [ntransf [type [ndim [Nmodes1 Nmodes2 Nmodes3 [Nsrc
                  [tol [debug [spread_scheme [spread_sort [upsampfac]]]]]]]]]]

   debug = 0: rel errors and overall timing
           1: timing breakdowns
           2: also spreading output
   
   spread_scheme = 0: sequential maximally multithreaded spread/interp
                   1: parallel singlethreaded spread/interp, nested last batch
   
   Example: finufftGuru_test 10 1 2 1000 1000 0 1000000 1e-12 0 [0 2 2.0]

   For Type3, Nmodes{1,2,3} controls the spread of NU freq targs in each dim.
   Example w/ nk = 5000: finufftGuru_test 1 3 2 100 50 0 1000000 1e-12 0

   By: Andrea Malleo 2019, tweaked by Alex Barnett 2020.
*/
{
  int ntransf = 1;  // defaults...
  int type = 1;
  int ndim = 1;
  BIGINT M = 1e6, N1 = 1000, N2 = 500, N3=250;   // M = # srcs, N1,N2,N3= # modes in each dim
  double tol = 1e-6;
  int optsDebug = 0, sprDebug = 0;
  int sprScheme = 0;
  int sprSort = 2;
  double upsampfac = 2.0;     // either 2.0 or 1.25
  int isign = +1;             // choose which exponential sign to test
  
  // Collect command line arguments ------------------------------------------
  if (argc>1) sscanf(argv[1],"%d",&ntransf);
  if(argc > 2) sscanf(argv[2],"%d",&type);
  if(argc > 3) sscanf(argv[3],"%d",&ndim);
  double w;
  if(argc > 4) {
    sscanf(argv[4],"%lf",&w); N1 = (BIGINT)w;
    sscanf(argv[5],"%lf",&w); N2 = (BIGINT)w;
    sscanf(argv[6],"%lf",&w); N3 = (BIGINT)w;
  }
  if (argc>7) { sscanf(argv[7],"%lf",&w); M = (BIGINT)w; }
  if (argc>8) {
    sscanf(argv[8],"%lf",&tol);
    if (tol<=0.0) { printf("tol must be positive!\n"); return 1; }
  }
  if (argc>9) sscanf(argv[9],"%d",&optsDebug);
  sprDebug = (optsDebug>1) ? 1 : 0;  // see output from spreader
  if (argc>10) sscanf(argv[10], "%d", &sprScheme); 
  if (argc>11) sscanf(argv[11],"%d",&sprSort);
  if (argc>12) sscanf(argv[12],"%lf",&upsampfac);
  if (argc==1 || argc>13) {
    fprintf(stderr,"Usage: finufftGuru_test [ntransf [type [ndim [N1 N2 N3 [Nsrc [tol [debug [spread_scheme [spread_sort [upsampfac]]]]]]]]]]\n");
    return 1;
  }

  // Allocate and initialize input --------------------------------------------
  
  cout << scientific << setprecision(15);
  N2 = (N2 == 0) ? 1 : N2;
  N3 = (N3 == 0) ? 1 : N3;
  
  BIGINT N = N1*N2*N3;
  
  FLT *x = (FLT *)malloc(sizeof(FLT)*M);        // NU pts x coords
  if(!x){
    fprintf(stderr, "failed malloc x coords\n");
    return 1;
  }

  FLT *y = NULL;
  FLT *z = NULL;
  if(ndim > 1){
    y = (FLT *)malloc(sizeof(FLT)*M);        // NU pts y coords
    if(!y){
      fprintf(stderr, "failed malloc y coords\n");
      free(x);
      return 1;
    }
  }

  if(ndim > 2){
    z = (FLT *)malloc(sizeof(FLT)*M);        // NU pts z coords
    if(!z){
      fprintf(stderr, "failed malloc z coords\n");
      free(x);
      if(y)
	free(y);
      return 1;
    }
  }

  
  FLT* s = NULL; 
  FLT* t = NULL; 
  FLT* u = NULL;

  if(type == 3){
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

  CPX* F = (CPX*)malloc(sizeof(CPX)*N*ntransf);   // mode ampls
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


  FFTW_CLEANUP();
  FFTW_CLEANUP_THREADS();
  FFTW_FORGET_WISDOM();
  //std::this_thread::sleep_for(std::chrono::seconds(1));
  sleep(1);


  // call FINUFFT -----------------------------------------------------------

  printf("------------------------GURU INTERFACE---------------------------\n");
  //Start by instantiating a finufft_plan
  finufft_plan plan;

  //then by instantiating a nufft_opts
  nufft_opts opts;
  
  //Guru Step 0
  finufft_default_opts(&opts);
  
  //Optional Customization opts 
  opts.debug = optsDebug;
  opts.spread_debug = sprDebug;
  plan.spopts.debug = sprDebug;
  opts.spread_sort = sprSort;
  opts.upsampfac = upsampfac;
  opts.spread_scheme = sprScheme;

  BIGINT n_modes[3];
  n_modes[0] = N1;
  n_modes[1] = N2;
  n_modes[2] = N3; //#modes per dimension 

  int blksize = MY_OMP_GET_MAX_THREADS(); 

  CNTime timer; timer.start();  
  //Guru Step 1
  int ier = finufft_makeplan(type, ndim,  n_modes, isign, ntransf, tol, blksize, &plan, &opts);
  //for type3, omit n_modes and send in NULL

  //the opts struct can no longer be modified with effect!
  
  double plan_t = timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
    return ier;
  } else{
    printf("plan creation for %lld modes: %.3g s\n", (long long)N, plan_t);
  }
  
  timer.restart();
  //Guru Step 2
  ier = finufft_setpts(&plan, M, x, y, z, N, s, t, u); //type 1+2, N=0, s,t,u = NULL
  double sort_t = timer.elapsedsec();
  if (ier) {
    printf("error (ier=%d)!\n",ier);
    return ier;
  } else{
    printf("setpts for %lld src points: %.3g s\n", (long long)M, sort_t);
  }
  
  timer.restart();
  //Guru Step 3
  ier = finufft_exec(&plan,c,F);
  double  exec_t=timer.elapsedsec();

  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
    return ier;
  } else
    printf("exec %d of %lld NU pts to %lld modes: %.3g s \t%.3g NU pts/s\n", ntransf, 
	   (long long)M,(long long)N, exec_t , ntransf*M/exec_t);

  //Guru Step 4
  timer.restart();
  finufft_destroy(&plan);
  double destroy_t = timer.elapsedsec();
  printf("finufft_destroy completed in %.3g s\n", destroy_t);
  //You're done!
  

  // Do a timing ratio against the simple interface ---------------------------

  double totalTime = plan_t + sort_t + exec_t + destroy_t;
  //comparing timing results with repeated calls to corresponding finufft function 

  FFTW_CLEANUP();
  FFTW_CLEANUP_THREADS();
  FFTW_FORGET_WISDOM();

  //std::this_thread::sleep_for(std::chrono::seconds(1)); if c++11 is allowed
  sleep(1); //sleep for one second using linux sleep call
  
  printf("------------------------SIMPLE INTERFACE-------------------------\n");
  // this used to actually call Alex's old (v1.1) src/finufft?d.cpp routines.
  // Since we don't want to ship those, we now call the simple interfaces.
  
  double oldTime = many_simple_calls(c,F, &plan);

  FFTW_CLEANUP();
  FFTW_CLEANUP_THREADS();
  FFTW_FORGET_WISDOM();
  //std::this_thread::sleep_for(std::chrono::seconds(1));
  sleep(1);
  printf("%d of\t%lld NU pts to %lld modes in %.3g s      \t%.3g NU pts/s\n",
         ntransf,(long long)M,(long long)N, oldTime , ntransf*M/oldTime);
  
  printf("\tspeedup: T_finufft%dd%d_simple / T_finufft%dd%d = %.3g\n",ndim,type,
         ndim, type, oldTime/totalTime);
  
  
  //--------------------------------------- Free Memory
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



// ---------------------------------------------------------------------------
double finufftFunnel(CPX *cStart, CPX *fStart, finufft_plan *plan)
// HELPER FOR COMPARING AGAINST SIMPLE INTERFACES. Reads opts from the
// finufft plan, and does a single simple interface call.
// returns the run-time in seconds, or -1.0 if error.
{

  CNTime timer; timer.start();
  int ier = 0;
  double t = 0;
  double fail = -1.0;
  nufft_opts* popts = &(plan->opts);   // opts ptr, as v1.2 simple calls need
  switch(plan->n_dims){

    /*1d*/
  case 1:
    switch(plan->type){

    case 1:
      timer.restart();
      ier = finufft1d1(plan->nj, plan->X, cStart, plan->fftsign, plan->tol, plan->ms, fStart, popts);
      t = timer.elapsedsec();
      if(ier)
	return fail;
      else
	return t;
      
    case 2:
      timer.restart();
      ier = finufft1d2(plan->nj, plan->X, cStart, plan->fftsign, plan->tol, plan->ms, fStart, popts);
      t = timer.elapsedsec();
      if(ier)
	return fail;
      else
	return t;
      
    case 3:
      timer.restart();
      ier = finufft1d3(plan->nj, plan->X_orig, cStart, plan->fftsign, plan->tol, plan->nk, plan->s, fStart, popts);
      t = timer.elapsedsec();
      if(ier)
	return fail;
      else
	return t;
      
    default:
      return fail; 

    }

    /*2d*/
  case 2:
    switch(plan->type){
      
    case 1:
      timer.restart();
      ier = finufft2d1(plan->nj, plan->X, plan->Y, cStart, plan->fftsign, plan->tol, plan->ms, plan->mt, fStart, popts);
      t = timer.elapsedsec();
      if(ier)
	return fail;
      else
	return t;
      
    case 2:
      timer.restart();
      ier = finufft2d2(plan->nj, plan->X, plan->Y, cStart, plan->fftsign, plan->tol, plan->ms, plan->mt,
     		       fStart, popts);
      t = timer.elapsedsec();
      if(ier)
	return fail;
      else
	return t;

    case 3:
      timer.restart();
      ier = finufft2d3(plan->nj, plan->X_orig, plan->Y_orig, cStart, plan->fftsign, plan->tol, plan->nk, plan->s, plan->t,
                       fStart, popts); 
      t = timer.elapsedsec();
      if(ier)
	return fail;
      else
	return t;
      
    default:
      return fail;
    }

    /*3d*/
  case 3:
    
    switch(plan->type){

    case 1:
      timer.restart();
      ier = finufft3d1(plan->nj, plan->X, plan->Y, plan->Z, cStart, plan->fftsign, plan->tol,
                       plan->ms, plan->mt, plan->mu, fStart, popts);
      t = timer.elapsedsec();
      if(ier)
	return fail;
      else
	return t;
      
    case 2:
      timer.restart();
      ier = finufft3d2(plan->nj, plan->X, plan->Y, plan->Z, cStart, plan->fftsign, plan->tol,
                       plan->ms, plan->mt, plan->mu, fStart, popts);
      t = timer.elapsedsec();
      if(ier)
	return fail;
      else
	return t;
      
    case 3:
      timer.restart();
      ier = finufft3d3(plan->nj, plan->X_orig, plan->Y_orig, plan->Z_orig, cStart, plan->fftsign, plan->tol,
                       plan->nk, plan->s, plan->t, plan->u, fStart, popts);
      t = timer.elapsedsec();
      if(ier)
	return fail;
      else
	return t;

    /*invalid type*/
    default:
      return fail;
    }

    /*invalid dimension*/
  default:
    return fail;
  }
}


double many_simple_calls(CPX *c,CPX *F,finufft_plan *plan)
// A unified interface to all of the simple interfaces
// (was actually runOldFinufft, calling the old v1.1 lib, which was in subdir)
{
    
    CPX *cStart;
    CPX *fStart;

    double time = 0;
    double temp = 0;;
    
    for(int k = 0; k < plan->n_transf; k++){
      cStart = c + plan->nj*k;
      fStart = F + plan->ms*plan->mt*plan->mu*k;
      
      /*if(k != 0){
	plan->opts.debug = 0;
	plan->opts.spread_debug = 0;
	}*/
      
      temp = finufftFunnel(cStart,fStart, plan);
      if(temp == -1.0){
	printf("Call to finufft FAILED!"); 
	time = -1.0;
	break;
      }
      else
	time += temp;
    }
    return time;
}
