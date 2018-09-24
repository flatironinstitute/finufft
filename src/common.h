#ifndef COMMON_H
#define COMMON_H

#include "finufft.h"
#include "defs.h"
#include "spreadinterp.h"
#include <fftw3.h>

// defs internal to common.cpp...
typedef std::complex<double> dcomplex;
// (slightly sneaky since name duplicated by mwrap - unrelated)


// common.cpp provides...
int setup_spreader_for_nufft(spread_opts &spopts, FLT eps, nufft_opts opts);
void set_nf_type12(BIGINT ms, nufft_opts opts, spread_opts spopts,BIGINT *nf);
void set_nhg_type3(FLT S, FLT X, nufft_opts opts, spread_opts spopts,
		  BIGINT *nf, FLT *h, FLT *gam);
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
#endif  // COMMON_H
