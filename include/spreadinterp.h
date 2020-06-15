// Defines interface to spreading/interpolation code.
// Note: see defs.h for definition of MAX_NSPREAD (as of 9/24/18).

#ifndef SPREADINTERP_H
#define SPREADINTERP_H

#include <dataTypes.h>
#include <spread_opts.h>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// NU coord handling macro: if p is true, rescales from [-pi,pi] to [0,N], then
// folds *only* one period below and above, ie [-N,2N], into the domain [0,N]...
#define RESCALE(x,N,p) (p ? \
		     ((x*M_1_2PI + (x<-PI ? 1.5 : (x>PI ? -0.5 : 0.5)))*N) : \
		     (x<0 ? x+N : (x>N ? x-N : x)))
// yuk! But this is *so* much faster than slow std::fmod that we stick to it.
// todo: replace w/ C++ function; apparently will be as fast


/* Bitwise debugging timing flag (TF) definitions; see spread_opts.flags.
    This is an unobtrusive way to determine the time contributions of the
    different components of spreading/interp by selectively leaving them out.
    For example, running the following two tests shows the effect of the exp()
    in the kernel evaluation (the last argument is the flag):
    > test/spreadtestnd 3 8e6 8e6 1e-6 1 0 0 1 0
    > test/spreadtestnd 3 8e6 8e6 1e-6 1 4 0 1 0
    NOTE: non-zero values are for experts only, since
    NUMERICAL OUTPUT MAY BE INCORRECT UNLESS spread_opts.flags=0 !
*/
#define TF_OMIT_WRITE_TO_GRID        1 // don't add subgrids to out grid (dir=1)
#define TF_OMIT_EVALUATE_KERNEL      2 // don't evaluate the kernel at all
#define TF_OMIT_EVALUATE_EXPONENTIAL 4 // omit exp() in kernel (kereval=0 only)
#define TF_OMIT_SPREADING            8 // don't interp/spread (dir=1: to subgrids)

// things external (spreadinterp) interface needs...
int spreadinterp(BIGINT N1, BIGINT N2, BIGINT N3, FLT *data_uniform,
		 BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		 FLT *data_nonuniform, spread_opts opts);
int spreadcheck(BIGINT N1, BIGINT N2, BIGINT N3,
                 BIGINT M, FLT *kx, FLT *ky, FLT *kz, spread_opts opts);
int indexSort(BIGINT* sort_indices, BIGINT N1, BIGINT N2, BIGINT N3, BIGINT M, 
               FLT *kx, FLT *ky, FLT *kz, spread_opts opts);
int interpSorted(BIGINT* sort_indices,BIGINT N1, BIGINT N2, BIGINT N3, 
		      FLT *data_uniform,BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		 FLT *data_nonuniform, spread_opts opts, int did_sort);
int spreadSorted(BIGINT* sort_indices,BIGINT N1, BIGINT N2, BIGINT N3, 
		      FLT *data_uniform,BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		 FLT *data_nonuniform, spread_opts opts, int did_sort);
int spreadinterpSorted(BIGINT* sort_indices,BIGINT N1, BIGINT N2, BIGINT N3, 
		      FLT *data_uniform,BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		      FLT *data_nonuniform, spread_opts opts, int did_sort);
FLT evaluate_kernel(FLT x,const spread_opts &opts);
FLT evaluate_kernel_noexp(FLT x,const spread_opts &opts);
int setup_spreader(spread_opts &opts,FLT eps,FLT upsampfac,int kerevalmeth, int debug, int showwarn);

#endif  // SPREADINTERP_H
