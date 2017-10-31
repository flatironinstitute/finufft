#ifndef COMMON_H
#define COMMON_H

#include "utils.h"
#include "finufft.h"
#include "cnufftspread.h"
#include <fftw3.h>

// constants needed within common
#define MAX_NQUAD 100              // max number of positive quadr nodes
// increase this if you have >1TB RAM...
#define MAX_NF    (BIGINT)1e11     // too big to ever succeed (next235 ~ 1s)

// common.cpp provides...
int setup_spreader_for_nufft(spread_opts &spopts, FLT eps, nufft_opts opts);
void set_nf_type12(BIGINT ms, nufft_opts opts, spread_opts spopts,INT64 *nf);
void set_nhg_type3(FLT S, FLT X, nufft_opts opts, spread_opts spopts,
		  INT64 *nf, FLT *h, FLT *gam);
void onedim_dct_kernel(BIGINT nf, FLT *fwkerhalf, spread_opts opts);
void onedim_fseries_kernel(BIGINT nf, FLT *fwkerhalf, spread_opts opts);
void onedim_nuft_kernel(BIGINT nk, FLT *k, FLT *phihat, spread_opts opts);
void deconvolveshuffle1d(int dir,FLT prefac,FLT* ker,BIGINT ms,FLT *fk,
			 BIGINT nf1,FFTW_CPX* fw,int modeord);
void deconvolveshuffle2d(int dir,FLT prefac,FLT *ker1, FLT *ker2,
			 BIGINT ms,BIGINT mt,
			 FLT *fk, BIGINT nf1, BIGINT nf2, FFTW_CPX* fw,
			 int modeord);
void deconvolveshuffle3d(int dir,FLT prefac,FLT *ker1, FLT *ker2,
			 FLT *ker3, BIGINT ms, BIGINT mt, BIGINT mu,
			 FLT *fk, BIGINT nf1, BIGINT nf2, BIGINT nf3,
			 FFTW_CPX* fw, int modeord);
#endif
