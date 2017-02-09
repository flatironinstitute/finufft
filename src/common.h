#ifndef COMMON_H
#define COMMON_H

#include "finufft.h"

BIGINT set_nf(BIGINT ms, nufft_opts opts, spread_opts spopts);
void onedim_dct_kernel(BIGINT nf, double *fwkerhalf,
		       double &prefac_unused_dims, spread_opts opts);
void onedim_fseries_kernel(BIGINT nf, double *fwkerhalf,
		       double &prefac_unused_dims, spread_opts opts);
void onedim_nuft_kernel(BIGINT nk, double *k, double *phihat,
			double &prefac_unused_dim, spread_opts opts);
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
