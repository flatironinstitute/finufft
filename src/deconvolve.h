#ifndef __DECONVOLVE_H__
#define __DECONVOLVE_H__

#include "spread.h"

__global__
void Deconvolve_2d(int ms, int mt, int nf1, int nf2, int fw_width, CUCPX* fw, CUCPX *fk,
                   FLT *fwkerhalf1, FLT *fwkerhalf2);

int cudeconvolve2d(spread_opts opts, cufinufft_plan *d_mem);
#endif
