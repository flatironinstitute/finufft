#ifndef COMMON_H
#define COMMON_H

#include "utils.h"
#include <fftw3.h>
#include "finufft.h"

// constants needed by common.cpp:
#define MAX_NQUAD 100     // max number of positive quadr nodes

// common.cpp provides...
void set_nf_type12(BIGINT ms, nufft_opts opts, spread_opts spopts,BIGINT *nf);
void set_nhg_type3(double S, double X, nufft_opts opts, spread_opts spopts,
		   BIGINT *nf, double *h, double *gam);
void onedim_dct_kernel(BIGINT nf, double *fwkerhalf, spread_opts opts);
void onedim_fseries_kernel(BIGINT nf, double *fwkerhalf, spread_opts opts);
void onedim_nuft_kernel(BIGINT nk, double *k, double *phihat, spread_opts opts);
void deconvolveshuffle1d(int dir,double prefac,double* ker,BIGINT ms,double *fk,
			 BIGINT nf1,fftw_complex* fw);
void deconvolveshuffle2d(int dir,double prefac,double *ker1, double *ker2,
			 BIGINT ms,BIGINT mt,
			 double *fk, BIGINT nf1, BIGINT nf2, fftw_complex* fw);
void deconvolveshuffle3d(int dir,double prefac,double *ker1, double *ker2,
			 double *ker3, BIGINT ms, BIGINT mt, BIGINT mu,
			 double *fk, BIGINT nf1, BIGINT nf2, BIGINT nf3,
			 fftw_complex* fw);
#endif
