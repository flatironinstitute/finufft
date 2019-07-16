// minimum definitions needed for interface to FINUFFT library, from C++ or C

#ifndef FINUFFT_H
#define FINUFFT_H

#include <dataTypes.h>
#include <nufft_opts.h>
#include <spreadinterp.h>
#include <fftw_defs.h>

enum finufft_type { type1, type2, type3};

typedef struct {

  FLT X1,C1,D1,h1,gam1;
  FLT X2,C2,D2,h2,gam2;
  FLT X3,C3,D3,h3,gam3;

} type3Params;


typedef struct {

  finufft_type type;
  int n_dims;
  int n_transf;
  int nj; 
  int nk;
  FLT tol;
  
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
  
}finufft_plan;



// ------------------ library provides ------------------------------------
#ifdef __cplusplus
extern "C"
{
#endif

// ------------------ Guru Interface ------------------------------------

int make_finufft_plan(finufft_type type, int n_dims, BIGINT* n_modes, int iflag, int n_transf, FLT tol, finufft_plan *plan );
void finufft_default_opts(nufft_opts *o);
int setNUpoints(finufft_plan * plan , BIGINT M, FLT *xj, FLT *yj, FLT *zj, BIGINT N, FLT *s, FLT *t, FLT *u); 
int finufft_exec(finufft_plan * plan ,  CPX *weights, CPX * result);
int finufft_destroy(finufft_plan * plan);


  
#ifdef __cplusplus
}
#endif


#endif   // FINUFFT_H
