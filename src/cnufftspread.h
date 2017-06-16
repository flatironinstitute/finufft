#ifndef CNUFFTSPREAD_H
#define CNUFFTSPREAD_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"

#define MAX_NSPREAD 16     // upper bound on w, ie nspread, also for common

// Note -std=c++11 is needed to avoid warning for static initialization here:
struct spread_opts {
  int nspread=6;           // w, the kernel width in grid pts
  // opts controlling spreading method (indep of kernel)...
  int spread_direction=1;  // 1 means spread NU->U, 2 means interpolate U->NU
  int pirange=0;           // 0: coords in [0,N), 1 coords in [-pi,pi)
  int chkbnds=1;           // 0: don't check NU pts in range; 1: do (may segfault)
  int sort=1;              // 0: don't sort NU pts, 1: do sort
  BIGINT max_subproblem_size=1e5; // extra RAM per thread
  int flags=0;             // binary flags for timing only (may give wrong ans!)
  int debug=0;             // 0: silent; 1: text output
  // ES kernel specific...
  FLT ES_beta;
  FLT ES_halfwidth;
  FLT ES_c;
};

// macro: if p is true, rescales from [-pi,pi] to [0,N] ...
#define RESCALE(x,N,p) (p ? ((x*M_1_2PI + 0.5)*N) : x)

/* Bitwise timing flag definitions; see spread_opts.timing_flags.
    This is an unobtrusive way to determine the time contributions of the different
    components of the algorithm by selectively leaving them out.
    For example, running the following two tests should show the modest gain
    achieved by bin-sorting the subproblems for dir=1 in 3D (the last argument is the
    flag):
    > test/spreadtestnd 3 1e7 1e6 1e-6 2 0
    > test/spreadtestnd 3 1e7 1e6 1e-6 2 16
*/
#define TF_OMIT_WRITE_TO_GRID          1  // don't write to the output grid at all
#define TF_OMIT_EVALUATE_KERNEL        2  // don't evaluate the kernel at all
#define TF_OMIT_EVALUATE_EXPONENTIAL   4  // don't evaluate the exp operation in the kernel
#define TF_OMIT_PI_RANGE               8  // don't convert the data to/from [-pi,pi) range
#define TF_OMIT_SORT_SUBPROBLEMS       16 // don't bin-sort the subproblems
#define TF_OMIT_SPREADING              32 // don't spread at all!


// things external interface needs...
int cnufftspread(BIGINT N1, BIGINT N2, BIGINT N3, FLT *data_uniform,
		 BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		 FLT *data_nonuniform, spread_opts opts);
FLT evaluate_kernel(FLT x,const spread_opts &opts);
FLT evaluate_kernel_noexp(FLT x,const spread_opts &opts);
int setup_kernel(spread_opts &opts,FLT eps,FLT R);

#endif // CNUFFTSPREAD_H
