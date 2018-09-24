// interface to spreading/interpolation code.
// Note: see defs.h for definition of MAX_NSPREAD (as of 9/24/18).

#ifndef SPREADINTERP_H
#define SPREADINTERP_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "finufft.h"
#include "defs.h"
#include "utils.h"

struct spread_opts {      // see cnufftspread:setup_spreader for defaults.
  int nspread;            // w, the kernel width in grid pts
  int spread_direction;   // 1 means spread NU->U, 2 means interpolate U->NU
  int pirange;            // 0: coords in [0,N), 1 coords in [-pi,pi)
  int chkbnds;            // 0: don't check NU pts are in range; 1: do
  int sort;               // 0: don't sort NU pts, 1: do, 2: heuristic choice
  int kerevalmeth;        // 0: exp(sqrt()), old, or 1: Horner ppval, fastest
  int kerpad;             // 0: no pad to mult of 4, 1: do (helps i7 kereval=0)
  int sort_threads;       // 0: auto-choice, >0: fix number of sort threads
  BIGINT max_subproblem_size; // sets extra RAM per thread
  int flags;              // binary flags for timing only (may give wrong ans!)
  int debug;              // 0: silent, 1: small text output, 2: verbose
  FLT upsampfac;          // sigma, upsampling factor, default 2.0
  // ES kernel specific...
  FLT ES_beta;
  FLT ES_halfwidth;
  FLT ES_c;
};

// NU coord handling macro: if p is true, rescales from [-pi,pi] to [0,N], then
// folds *only* one period below and above, ie [-N,2N], into the domain [0,N]...
#define RESCALE(x,N,p) (p ? \
		     ((x*M_1_2PI + (x<-PI ? 1.5 : (x>PI ? -0.5 : 0.5)))*N) : \
		     (x<0 ? x+N : (x>N ? x-N : x)))
// yuk! But this is *so* much faster than slow std::fmod that we stick to it.


/* Bitwise timing flag definitions; see spread_opts.flags.
    This is an unobtrusive way to determine the time contributions of the
    different components of spreading/interp by selectively leaving them out.
    For example, running the following two tests shows the effect of the exp()
    in the kernel evaluation (the last argument is the flag):
    > test/spreadtestnd 3 8e6 8e6 1e-6 1 0 0 1 0
    > test/spreadtestnd 3 8e6 8e6 1e-6 1 4 0 1 0
    NOTE: NUMERICAL OUTPUT MAY BE INCORRECT UNLESS spread_opts.flags=0 !
*/
#define TF_OMIT_WRITE_TO_GRID        1 // don't add subgrids to out grid (dir=1)
#define TF_OMIT_EVALUATE_KERNEL      2 // don't evaluate the kernel at all
#define TF_OMIT_EVALUATE_EXPONENTIAL 4 // omit exp() in kernel (kereval=0 only)
#define TF_OMIT_SPREADING            8 // don't interp/spread (dir=1: to subgrids)


// things external interface needs...
int spreadinterp(BIGINT N1, BIGINT N2, BIGINT N3, FLT *data_uniform,
		 BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		 FLT *data_nonuniform, spread_opts opts);

int spreadcheck(BIGINT N1, BIGINT N2, BIGINT N3,
                 BIGINT M, FLT *kx, FLT *ky, FLT *kz,
                 spread_opts opts);

int spreadsort(BIGINT* sort_indices, BIGINT N1, BIGINT N2, BIGINT N3, BIGINT M, 
               FLT *kx, FLT *ky, FLT *kz, spread_opts opts);

int spreadwithsortidx(BIGINT* sort_indices,BIGINT N1, BIGINT N2, BIGINT N3, 
		      FLT *data_uniform,BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		      FLT *data_nonuniform, spread_opts opts, int did_sort);

FLT evaluate_kernel(FLT x,const spread_opts &opts);
FLT evaluate_kernel_noexp(FLT x,const spread_opts &opts);
int setup_spreader(spread_opts &opts,FLT eps,FLT upsampfac,int kerevalmeth);

#endif  // SPREADINTERP_H
