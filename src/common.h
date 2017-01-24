#ifndef COMMON_H
#define COMMON_H

#include "finufft1d.h"

void onedim_dct_kernel(BIGINT nf, double *fwkerhalf,
		       double &prefac_unused_dims, spread_opts opts);

#endif
