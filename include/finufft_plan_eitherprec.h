// Switchable-precision interface template for FINUFFT PLAN struct, used by
// finufft_plan.h.
// Internal use only: users should link to finufft.h
// Barnett 7/5/20

#if (!defined(FINUFFT_PLAN_H) && !defined(SINGLE)) || (!defined(FINUFFTF_PLAN_H) && defined(SINGLE))
// Make sure we don't include double and single headers more than once each...
#ifndef SINGLE
#define FINUFFT_PLAN_H
#else
#define FINUFFTF_PLAN_H
#endif

#include <fftw_defs.h>
#include <nufft_opts.h>
#include <spread_opts.h>

#ifndef __cplusplus
#include <stdbool.h>     // for bools in C
#endif

// clear macros so can refine
#undef TYPE3PARAMS
#undef FINUFFT_PLAN
#undef FINUFFT_PLAN_S
#ifdef SINGLE
#define FINUFFT_PLAN_S finufftf_plan_s
#define TYPE3PARAMS type3Paramsf
#define FINUFFT_PLAN finufftf_plan
#else
#define FINUFFT_PLAN_S finufft_plan_s
#define TYPE3PARAMS type3Params
#define FINUFFT_PLAN finufft_plan
#endif

// the plan handle that we pass around is just a pointer to the struct that
// contains all the info
typedef struct FINUFFT_PLAN_S * FINUFFT_PLAN;

// group together a bunch of type 3 rescaling/centering/phasing parameters:
typedef struct {
  FLT X1,C1,D1,h1,gam1;  // x dim: X=halfwid C=center D=freqcen h,gam=rescale
  FLT X2,C2,D2,h2,gam2;  // y
  FLT X3,C3,D3,h3,gam3;  // z
} TYPE3PARAMS;


typedef struct FINUFFT_PLAN_S {  // the main plan struct; note C-compatible struct
  
  int type;        // transform type (Rokhlin naming): 1,2 or 3
  int dim;         // overall dimension: 1,2 or 3
  int ntrans;      // how many transforms to do at once (vector or "many" mode)
  int nj;          // number of NU pts in type 1,2 (for type 3, num input x pts)
  int nk;          // number of NU freq pts (type 3 only)
  FLT tol;         // relative user tolerance
  int batchSize;   // # strength vectors to group together for FFTW, etc
  int nbatch;      // how many batches done to cover all ntrans vectors
  
  BIGINT ms;       // number of modes in x (1) dir (historical CMCL name) = N1
  BIGINT mt;       // number of modes in y (2) direction = N2
  BIGINT mu;       // number of modes in z (3) direction = N3
  BIGINT N;        // total # modes (prod of above three)
  
  BIGINT nf1;      // size of internal fine grid in x (1) direction
  BIGINT nf2;      // " y
  BIGINT nf3;      // " z
  BIGINT nf;       // total # fine grid points (product of the above three)
  
  int fftSign;     // sign in exponential for NUFFT defn, guaranteed to be +-1

  FLT* phiHat1;    // FT of kernel in t1,2, on x-axis mode grid
  FLT* phiHat2;    // " y-axis.
  FLT* phiHat3;    // " z-axis.
  
  FFTW_CPX* fwBatch;    // (batches of) fine grid(s) for FFTW to plan & act on.
                        // Usually the largest working array
  
  BIGINT *sortIndices;  // precomputed NU pt permutation, speeds spread/interp
  bool didSort;         // whether binsorting used (false: identity perm used)

  FLT *X, *Y, *Z;  // for t1,2: ptr to user-supplied NU pts (no new allocs).
                   // for t3: allocated as "primed" (scaled) src pts x'_j, etc

  // type 3 specific
  FLT *S, *T, *U;  // pointers to user's target NU pts arrays (no new allocs)
  CPX* prephase;   // pre-phase, for all input NU pts
  CPX* deconv;     // reciprocal of kernel FT, phase, all output NU pts
  CPX* CpBatch;    // working array of prephased strengths
  FLT *Sp, *Tp, *Up;  // internal primed targs (s'_k, etc), allocated
  TYPE3PARAMS t3P; // groups together type 3 shift, scale, phase, parameters
  FINUFFT_PLAN innerT2plan;   // ptr used for type 2 in step 2 of type 3
  
  // other internal structs; each is C-compatible of course
  FFTW_PLAN fftwPlan;
  nufft_opts opts;     // this and spopts could be made ptrs
  spread_opts spopts;
  
} FINUFFT_PLAN_S;

#endif  // FINUFFT_PLAN_H or FINUFFTF_PLAN_H
