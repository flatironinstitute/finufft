#ifndef COMMON_H
#define COMMON_H

#include "utils.h"
#include "spreadinterp.h"

// constants needed within common
#define MAX_NQUAD 100              // max number of positive quadr nodes
// increase this if you need >1TB RAM...
#define MAX_NF    (BIGINT)1e11     // too big to ever succeed (next235 takes 1s)

struct cufinufft_opts;

// common.cpp provides...
int setup_spreader_for_nufft(spread_opts &spopts, FLT eps, cufinufft_opts opts);
void set_nf_type12(BIGINT ms, cufinufft_opts opts, spread_opts spopts,BIGINT *nf);
void onedim_fseries_kernel(BIGINT nf, FLT *fwkerhalf, spread_opts opts);
#endif  // COMMON_H
