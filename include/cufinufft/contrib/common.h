#ifndef COMMON_H
#define COMMON_H

#include "dataTypes.h"
#include "utils.h"
#include "utils_fp.h"
#include "spreadinterp.h"

// constants needed within common
#define MAX_NQUAD 100              // max number of positive quadr nodes
// increase this if you need >1TB RAM...
#define MAX_NF    (CUFINUFFT_BIGINT)INT_MAX  // In cufinufft we limit array sizes to 2^31
                                   // which is about 2 billion, since we set
                                   // CUFINUFFT_BIGINT to int. (Differs from FINUFFT)

struct cufinufft_opts;

// common.cpp provides...
int setup_spreader_for_nufft(SPREAD_OPTS &spopts, CUFINUFFT_FLT eps, cufinufft_opts opts);
void SET_NF_TYPE12(CUFINUFFT_BIGINT ms, cufinufft_opts opts, SPREAD_OPTS spopts,CUFINUFFT_BIGINT *nf,
                   CUFINUFFT_BIGINT b);
void onedim_fseries_kernel(CUFINUFFT_BIGINT nf, CUFINUFFT_FLT *fwkerhalf, SPREAD_OPTS opts);
void onedim_fseries_kernel_precomp(CUFINUFFT_BIGINT nf, CUFINUFFT_FLT *f, dcomplex *a, SPREAD_OPTS opts);
void onedim_fseries_kernel_compute(CUFINUFFT_BIGINT nf, CUFINUFFT_FLT *f, dcomplex *a, CUFINUFFT_FLT *fwkerhalf, SPREAD_OPTS opts);
#endif  // COMMON_H
