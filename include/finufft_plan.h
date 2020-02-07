#ifndef FINUFFT_PLAN_H
#define FINUFFT_PLAN_H

#include <fftw_defs.h>
#include <nufft_opts.h>
#include <spreadinterp.h>

//enum finufft_type { type1, type2, type3};


typedef struct {

  FLT X1,C1,D1,h1,gam1;
  FLT X2,C2,D2,h2,gam2;
  FLT X3,C3,D3,h3,gam3;

} type3Params;


typedef struct finufft_plan{
  //finufft_plan(){}

  int type;
  int n_dims;
  int n_transf;
  int nj; 
  int nk;
  FLT tol;
  int threadBlkSize;
  
  BIGINT ms;
  BIGINT mt;
  BIGINT mu;
  
  BIGINT nf1;
  BIGINT nf2;
  BIGINT nf3; 
  
  int iflag; 

  FLT * phiHat; //fourier coefficients of spreading kernel for all dims
  FFTW_CPX * fw; //fourier coefficients for all dims
  
  BIGINT *sortIndices; 
  bool didSort;

  //target freqs
  //type 3 only
  FLT * s; 
  FLT * t; 
  FLT * u;
  FLT * sp; 
  FLT * tp; 
  FLT * up; 

  FLT *X;
  FLT *Y;
  FLT *Z; 
  FLT *X_orig;
  FLT *Y_orig;
  FLT *Z_orig; 
  
  fftw_plan fftwPlan;
  
  nufft_opts opts;
  spread_opts spopts;
  type3Params t3P;

  bool isInnerT2;
  
}finufft_plan;



#endif

