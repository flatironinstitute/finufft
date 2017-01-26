#ifndef COMMON_H
#define COMMON_H

#include "finufft.h"

BIGINT set_nf(BIGINT ms, nufft_opts opts, spread_opts spopts);
void onedim_dct_kernel(BIGINT nf, double *fwkerhalf,
		       double &prefac_unused_dims, spread_opts opts);
void deconvolveshuffle1d(int dir,double prefac,double* ker,BIGINT ms,double *fk,
			 BIGINT nf1,fftw_complex* fw);

#endif
