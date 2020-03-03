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
  
  int type;        // transform type (Rokhlin naming): 1,2 or 3
  int dim;         // overall dimension: 1,2 or 3
  int n_transf;    // how many transforms to do at once (vector or "many" mode)
  int nj;          // number of NU pts (for type 3, the input x pts)
  int nk;          // number of NU freq pts (type 3 only)
  FLT tol;         // tolerance
  int batchSize;   // # strength vectors to group together for FFTW, etc
  
  BIGINT ms;       // number of modes in x (1) direction; old CMCL notation
  BIGINT mt;       // number of modes in y (2) direction
  BIGINT mu;       // number of modes in z (3) direction
  
  BIGINT nf1;      // size of internal fine grid in x (1) direction
  BIGINT nf2;      // " y
  BIGINT nf3;      // " z
  BIGINT nf;       // total fine grid points (product of the above 3)
  
  int fftSign;     // guaranteed to be +-1

  FLT* phiHat1;    // FT of kernel in t1,2, x-axis; for t3 it's all nk targs.
  FLT* phiHat2;    // " y
  FLT* phiHat3;    // " z
  
  FFTW_CPX* fwBatch;    // (batches of) fine grid(s) for FFTW to plan and act on
  
  BIGINT *sortIndices;  // precomputed NU x permutation, speeds spread/interp
  bool didSort;         // whether binsorting used (false: identity perm used)

  FLT *X;         // pointers to user-supplied NU pts arrays
  FLT *Y;
  FLT *Z; 

  // other internal structs; each is C-compatible of course
  FFTW_PLAN fftwPlan;
  nufft_opts opts;
  spread_opts spopts;
  type3Params t3P;

  // whether this plan is the type-2 inner call needed within a type-3 transform
  struct finufft_plan *innerT2Plan;   // used for type-2 as step 2 of type-3
  
} finufft_plan;


#endif  // FINUFFT_PLAN_H
