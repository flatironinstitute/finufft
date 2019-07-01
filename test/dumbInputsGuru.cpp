#include <finufft.h>
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


int main(int argc, char* argv[]){


  /*
    Calling the FINUFFT GURU library from C++ using all manner of crazy inputs
    that might cause errors. All should be caught gracefully.

    Compile with:
    
    Usage: ./dumbInputsGuru

   */


  BIGINT M = 1e6, N1 = 1000, N2 = 500;  // defaults: M = # srcs, N1,N2 = # modes
  double tol = 1e-6;          // default
  int isign = +1;             // choose which exponential sign to test
  int nvecs = 1;
  
  //at the moment this only goes to 2D
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
    for(BIGINT i = 0; i<nvecs*M; i++)
	c[i] = crandm11r(&se);
  }


  finufft_plan plan;

  BIGINT n_modes[3] {N1, N2, 1}; //#modes per dimension 

  int n_dims = 2;
  CNTime timer; timer.start();

  printf("2D dumb case ------------------------------------\n");
  int ier = make_finufft_plan(type1, n_dims,  n_modes, isign, nvecs, 0, &plan);
  printf("2D1, tol=0:\tier=%d (should complain)\n", ier);


  /*Type 1*/
  BIGINT no_modes[3]{0,0,0};
  ier = make_finufft_plan(type1, n_dims,  no_modes, isign, nvecs, tol, &plan);
  if(!ier)
    ier = setNUpoints(&plan, M, x, y, NULL, NULL);
  if(!ier)
      ier = finufft_exec(&plan,c,F);
  if(!ier)
    ier = finufft_destroy(&plan); 
  printf("2d1 Ns=Nt=0:\tier=%d \n", ier);

  BIGINT weird_modes[3]{0,N2,0};
  ier = make_finufft_plan(type1, n_dims,  weird_modes, isign, nvecs, tol, &plan);
  if(!ier)
    ier = setNUpoints(&plan, M, x, y, NULL, NULL);
  if(!ier)
      ier = finufft_exec(&plan,c,F);
  if(!ier)
    ier = finufft_destroy(&plan);
  printf("2D1 Ns=0,Nt>0:\tier=%d\n", ier );
  
  weird_modes[0] = N1;
  weird_modes[1] = 0;
  ier = make_finufft_plan(type1, n_dims,  weird_modes, isign, nvecs, tol, &plan);
  if(!ier)
    ier = setNUpoints(&plan, M, x, y, NULL, NULL);
  if(!ier)
      ier = finufft_exec(&plan,c,F);
  if(!ier)
    ier = finufft_destroy(&plan);
  printf("2D1 Ns>0,Nt=0:\tier=%d\n", ier );

  ier = make_finufft_plan(type1, n_dims,  n_modes, isign, nvecs, tol, &plan);
  int no_srcPts = 0;
  if(!ier)
    ier = setNUpoints(&plan, no_srcPts, x, y, NULL, NULL);
  if(!ier)
    ier = finufft_exec(&plan,c,F);
  if(!ier)
    ier = finufft_destroy(&plan);
  printf("2d1 M=0:\tier=%d\tnrm(F)=%.3g (should vanish) \n",ier,twonorm(N,F));


  /*Type 2*/
  for(int k = 0; k < N; k++) F[k] = sin((FLT)0.7*k) + IMA*cos((FLT)0.3*k); //set F for t2
  ier = make_finufft_plan(type2, n_dims, n_modes, isign, nvecs, 0, &plan);
  if(!ier)
    ier = setNUpoints(&plan, M, x, y, NULL, NULL);
  if(!ier)
      ier = finufft_exec(&plan,c,F);
  if(!ier)
    ier = finufft_destroy(&plan);
  printf("2d2 tol=0:\tier=%d (should complain)\n", ier);

  
  ier = make_finufft_plan(type2, n_dims,  no_modes, isign, nvecs, tol, &plan);
  if(!ier)
    ier = setNUpoints(&plan, M, x, y, NULL, NULL);
  if(!ier)
      ier = finufft_exec(&plan,c,F);
  if(!ier)
    ier = finufft_destroy(&plan); 
  printf("2d2 Ns=Nt=0:\tier=%d\tnrm(c)=%.3g (should vanish)\n", ier,twonorm(M,c));

  weird_modes[0] = 0;
  weird_modes[1] = N2;
  ier = make_finufft_plan(type2, n_dims,  weird_modes, isign, nvecs, tol, &plan);
  if(!ier)
    ier = setNUpoints(&plan, M, x, y, NULL, NULL);
  if(!ier)
      ier = finufft_exec(&plan,c,F);
  if(!ier)
    ier = finufft_destroy(&plan);
  printf("2D2 Ns=0,Nt>0:\tier=%d\tnrm(c)=%.3g (should vanish)\n", ier,twonorm(M,c) );
  
  weird_modes[0] = N1;
  weird_modes[1] = 0;
  ier = make_finufft_plan(type2, n_dims,  weird_modes, isign, nvecs, tol, &plan);
  if(!ier)
    ier = setNUpoints(&plan, M, x, y, NULL, NULL);
  if(!ier)
      ier = finufft_exec(&plan,c,F);
  if(!ier)
    ier = finufft_destroy(&plan);
  printf("2D2 Ns>0,Nt=0:\tier=%d\tnrm(c)=%.3g (should vanish)\n", ier,twonorm(M,c));

  ier = make_finufft_plan(type2, n_dims,  n_modes, isign, nvecs, tol, &plan);
  if(!ier)
    ier = setNUpoints(&plan, no_srcPts, x, y, NULL, NULL);
  if(!ier)
    ier = finufft_exec(&plan,c,F);
  if(!ier)
    ier = finufft_destroy(&plan);
  printf("2d2 M=0:\tier=%d\n",ier);
  
  
  
}
