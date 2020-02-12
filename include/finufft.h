// minimum definitions needed for interface to FINUFFT library, from C++ or C

#ifndef FINUFFT_H
#define FINUFFT_H

#include <dataTypes.h>
#include <nufft_opts.h>
#include <finufft_plan.h>

// ------------------ library provides ------------------------------------
#ifdef __cplusplus
extern "C"
{
#endif

// ------------------ Guru Interface ------------------------------------

  int finufft_makeplan(int type, int n_dims, BIGINT* n_modes, int iflag, int n_transf, FLT tol, int blksize, finufft_plan *plan, nufft_opts *o);
void finufft_default_opts(nufft_opts *o);
int finufft_setpts(finufft_plan * plan , BIGINT M, FLT *xj, FLT *yj, FLT *zj, BIGINT N, FLT *s, FLT *t, FLT *u); 
int finufft_exec(finufft_plan * plan ,  CPX *weights, CPX * result);
int finufft_destroy(finufft_plan * plan);


  
#ifdef __cplusplus
}
#endif


#endif   // FINUFFT_H
