#ifndef FINUFFT_PLAN_H
#define FINUFFT_PLAN_H

#include <fftw_defs.h>
#include <nufft_opts.h>
#include <spread_opts.h>

#ifndef __cplusplus
#include <stdbool.h>     // for bools in C
#endif

typedef struct {
  // groups together a bunch of type 3 rescaling/centering/phasing parameters
  FLT X1,C1,D1,h1,gam1;   // x dim
  FLT X2,C2,D2,h2,gam2;   // y
  FLT X3,C3,D3,h3,gam3;   // z
} type3Params;


typedef struct finufft_plan{  // the main plan object; note C-compatible struct
  
  int type;        // 1,2 or 3
  int n_dims;      // 1,2 or 3
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
  
  int fftsign;     // guaranteed to be +-1

  FLT * phiHat;    // FT of kernel (for each dim in t1,2; for nk targs in t3)
  FFTW_CPX * fw;   // (batches of) fine grid(s) for FFTW to act on
  
  BIGINT *sortIndices;  // precomputed NU x permutation, speeds spread/interp
  bool didSort;         // whether binsorting used (false: identity perm used)

  FLT * s;         // *** TO DELETE WHEN FIX t3
  FLT * t; 
  FLT * u; 
  FLT * sp;         // rescaled target freqs (relevant for type 3 only)
  FLT * tp; 
  FLT * up; 

  FLT *X;         // pointers to user-supplied NU pts arrays
  FLT *Y;
  FLT *Z; 
  FLT *X_orig;    // needed for t3 only
  FLT *Y_orig;
  FLT *Z_orig; 

  // other internal structs; each is C-compatible of course.
  FFTW_PLAN fftwPlan;  
  nufft_opts opts;
  spread_opts spopts;
  type3Params t3P;

  // whether this plan is the type-2 inner call needed within a type-3 transform
  bool isInnerT2;
  
} finufft_plan;


#endif  // FINUFFT_PLAN_H
