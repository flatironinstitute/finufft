#ifndef FINUFFT_PLAN_H
#define FINUFFT_PLAN_H

#include <fftw_defs.h>
#include <nufft_opts.h>
#include <spread_opts.h>

#ifndef __cplusplus
#include <stdbool.h>     // for bools in C
#endif

// group together a bunch of type-3 rescaling/centering/phasing parameters:
typedef struct {
  FLT X1,C1,D1,h1,gam1;   // x dim
  FLT X2,C2,D2,h2,gam2;   // y
  FLT X3,C3,D3,h3,gam3;   // z
} type3Params;


typedef struct finufft_plan{  // the main plan object; note C-compatible struct
  
  int type;        // transform type (Rokhlin naming): 1,2 or 3
  int dim;         // overall dimension: 1,2 or 3
  int ntrans;      // how many transforms to do at once (vector or "many" mode)
  int nj;          // number of NU pts in type 1,2 (for type 3, num input x pts)
  int nk;          // number of NU freq pts (type 3 only)
  FLT tol;         // relative tolerance
  int batchSize;   // # strength vectors to group together for FFTW, etc
  
  BIGINT ms;       // number of modes in x (1) direction (historical CMCL name)
  BIGINT mt;       // number of modes in y (2) direction
  BIGINT mu;       // number of modes in z (3) direction
  BIGINT N;        // total # modes
  
  BIGINT nf1;      // size of internal fine grid in x (1) direction
  BIGINT nf2;      // " y
  BIGINT nf3;      // " z
  BIGINT nf;       // total # fine grid points (product of the above three)
  
  int fftSign;     // sign in exponential for NUFFT defn, guaranteed to be +-1

  FLT* phiHat1;    // FT of kernel in t1,2, x-axis; for t3 it's all nk targs.
  FLT* phiHat2;    // " y-axis.
  FLT* phiHat3;    // " z-axis.
  
  FFTW_CPX* fwBatch;    // (batches of) fine grid(s) for FFTW to plan & act on.
                        // Usually the largest working array
  
  BIGINT *sortIndices;  // precomputed NU pt permutation, speeds spread/interp
  bool didSort;         // whether binsorting used (false: identity perm used)

  FLT *X;         // pointers to user-supplied NU pts arrays
  FLT *Y;
  FLT *Z; 

  FLT *X_orig;   // *** poss to delete, there for test guru t3
  FLT *Y_orig;
  FLT *Z_orig;
  FLT *s, *t, *u; // ***

  
  // other internal structs; each is C-compatible of course
  FFTW_PLAN fftwPlan;   // should these be ptrs to structs?
  nufft_opts opts;
  spread_opts spopts;
  type3Params t3P;       // groups together type 3 parameters

  // whether this plan is the type-2 inner call needed within a type-3 transform
  struct finufft_plan *innerT2Plan;   // used for type-2 as step 2 of type-3
  
} finufft_plan;


#endif  // FINUFFT_PLAN_H
