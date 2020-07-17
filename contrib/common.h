#ifndef COMMON_H
#define COMMON_H

#include "dataTypes.h"
#include "utils.h"
#include "utils_fp.h"
#include "spreadinterp.h"

// constants needed within common
#define MAX_NQUAD 100              // max number of positive quadr nodes
// increase this if you need >1TB RAM...
#define MAX_NF    (BIGINT)1e11     // too big to ever succeed (next235 takes 1s)

struct cufinufft_opts;

// common.cpp provides...
int setup_spreader_for_nufft(SPREAD_OPTS &spopts, FLT eps, cufinufft_opts opts);
void SET_NF_TYPE12(BIGINT ms, cufinufft_opts opts, SPREAD_OPTS spopts,BIGINT *nf,
                   BIGINT b);
void onedim_fseries_kernel(BIGINT nf, FLT *fwkerhalf, SPREAD_OPTS opts);
#endif  // COMMON_H
