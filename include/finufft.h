// minimum definitions needed for interface to FINUFFT library, from C++ or C

#ifdef T

#include <nufft_opts.h>
#include <fftw_defs.h>
#include <templates.h>
#include <defs.h>

#ifndef ONCE_FTYPE
#define ONCE_FTYPE
enum finufft_type { type1, type2, type3};
#endif

typedef struct {

  T X1,C1,D1,h1,gam1;
  T X2,C2,D2,h2,gam2;
  T X3,C3,D3,h3,gam3;

} TEMPLATE(type3Params,T) ;


typedef struct  {

  finufft_type type;
  int n_dims;
  int n_transf;
  int nj; 
  int nk;
  T tol;
  int threadBlkSize;
  
  BIGINT ms;
  BIGINT mt;
  BIGINT mu;
  
  BIGINT nf1;
  BIGINT nf2;
  BIGINT nf3; 
  
  int iflag; 

  T * phiHat; //fourier coefficients of spreading kernel for all dims
  TEMPLATE(FFTW_CPX,T) * fw; //fourier coefficients for all dims
  
  BIGINT *sortIndices; 
  bool didSort;

  //target freqs
  //type 3 only
  T * s; 
  T * t; 
  T * u;
  T * sp; 
  T * tp; 
  T * up; 

  T *X;
  T *Y;
  T *Z; 
  T *X_orig;
  T *Y_orig;
  T *Z_orig; 
  
  TEMPLATE(FFTW_PLAN,T) fftwPlan;
  
  nufft_opts opts;
  TEMPLATE(spread_opts,T) spopts;
  TEMPLATE(type3Params,T) t3P;

  bool isInnerT2;
  
} TEMPLATE(finufft_plan,T) ;


// ------------------ library provides ------------------------------------
#ifdef __cplusplus
extern "C"
{
#endif

// ------------------ Guru Interface ------------------------------------

  int TEMPLATE(make_finufft_plan,T)(finufft_type type, int n_dims, BIGINT* n_modes, int iflag, int n_transf, T tol, int blksize, TEMPLATE(finufft_plan,T) *plan );
  void finufft_default_opts(nufft_opts *o);
  int TEMPLATE(setNUpoints,T)(TEMPLATE(finufft_plan,T) * plan , BIGINT M, T *xj, T *yj, T *zj, BIGINT N, T *s, T *t, T *u); 
  int TEMPLATE(finufft_exec,T)(TEMPLATE(finufft_plan,T) * plan ,  TEMPLATE(CPX,T) *weights, TEMPLATE(CPX,T) * result);
  int TEMPLATE(finufft_destroy,T)(TEMPLATE(finufft_plan,T) * plan);


  
#ifdef __cplusplus
}
#endif

#endif   // def t

