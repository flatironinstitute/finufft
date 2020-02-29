#ifndef FINUFFT_PLAN_H
#define FINUFFT_PLAN_H

#include <fftw_defs.h>
#include <nufft_opts.h>
#include <spreadinterp.h>

typedef struct {
  // groups together a bunch of type 3 parameters, will live inside the finufft_plan
  FLT X1,C1,D1,h1,gam1;
  FLT X2,C2,D2,h2,gam2;
  FLT X3,C3,D3,h3,gam3;
} type3Params;


typedef struct finufft_plan{
  
  int type;        // 1,2 or 3
  int n_dims;      // 1,2, or 3
  int n_transf;    // how many transforms to do at once (vector or "many" mode)
  int nj;          // number of NU pts (for type 3, the input x pts)
  int nk;          // number of NU freq pts (type 3 only)
  FLT tol;         // tolerance
  int threadBlkSize;   // chunk size for vector "many" mode... I think
  
  BIGINT ms;        // number of modes in x (1) direction; old CMCL notation
  BIGINT mt;        // number of modes in y (2) direction
  BIGINT mu;        // number of modes in z (3) direction
  
  BIGINT nf1;       // size of internal fine grid in x (1) direction, etc
  BIGINT nf2;
  BIGINT nf3; 
  
  int iflag;

  FLT * phiHat;    // fourier coefficients of spreading kernel for all dims
  FFTW_CPX * fw;   // fourier coefficients for all dims
  
  BIGINT *sortIndices; 
  bool didSort;

  //target freqs (used at planning stage for type 3 only)
  FLT * s; 
  FLT * t; 
  FLT * u;
  FLT * sp; 
  FLT * tp; 
  FLT * up; 

  // NU point arrays
  FLT *X;
  FLT *Y;
  FLT *Z; 
  FLT *X_orig;
  FLT *Y_orig;
  FLT *Z_orig; 

  // other internal structs
  FFTW_PLAN fftwPlan;  
  nufft_opts opts;
  spread_opts spopts;
  type3Params t3P;

  // whether this plan is the type-2 inner call needed within a type-3 transform
  bool isInnerT2;
  
} finufft_plan;

#endif
