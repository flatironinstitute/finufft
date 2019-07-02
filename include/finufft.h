// minimum definitions needed for interface to FINUFFT library, from C++ or C

#ifndef FINUFFT_H
#define FINUFFT_H

#include <dataTypes.h>
#include <nufft_opts.h>
#include <spreadinterp.h>
#include <fftw_defs.h>

enum finufft_type { type1, type2, type3};

typedef struct {

  finufft_type type;
  int n_dims;
  int n_transf;
  int M; 

  BIGINT ms;
  BIGINT mt;
  BIGINT mu;
  
  BIGINT nf1;
  BIGINT nf2;
  BIGINT nf3; 
  
  int iflag; 

  FLT * fwker; //fourier coefficients of spreading kernel for all dims
  FFTW_CPX * fw; //fourier coefficients for all dims
  
  BIGINT *sortIndices; 
  bool didSort;
  
  FLT * targetFreqs; //type 3 only 

  FLT *X;
  FLT *Y;
  FLT *Z; 
  
  fftw_plan fftwPlan;
  
  nufft_opts opts;
  spread_opts spopts;
}finufft_plan;



// ------------------ library provides ------------------------------------
#ifdef __cplusplus
extern "C"
{
#endif

// ------------------ Guru Interface ------------------------------------

int make_finufft_plan(finufft_type type, int n_dims, BIGINT* n_modes, int iflag, int n_transf, FLT tol, finufft_plan *plan );
void finufft_default_opts(nufft_opts *o);
int setNUpoints(finufft_plan * plan , BIGINT M, FLT *Xpts, FLT *Ypts, FLT *Zpts, CPX *targetFreqs); 
int finufft_exec(finufft_plan * plan ,  CPX *weights, CPX * result);
int finufft_destroy(finufft_plan * plan);


  
#ifdef __cplusplus
}
#endif


#endif   // FINUFFT_H
