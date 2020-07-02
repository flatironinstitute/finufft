#ifndef __MEMTRANSFER_H__
#define __MEMTRANSFER_H__

#include <cufinufft.h>

int ALLOCGPUMEM1D_PLAN(cufinufft_plan *d_plan);
int ALLOCGPUMEM1D_NUPTS(cufinufft_plan *d_plan);
void FREEGPUMEMORY1D(cufinufft_plan *d_plan);

int ALLOCGPUMEM2D_PLAN(cufinufft_plan *d_plan);
int ALLOCGPUMEM2D_NUPTS(cufinufft_plan *d_plan);
void FREEGPUMEMORY2D(cufinufft_plan *d_plan);

int ALLOCGPUMEM3D_PLAN(cufinufft_plan *d_plan);
int ALLOCGPUMEM3D_NUPTS(cufinufft_plan *d_plan);
void FREEGPUMEMORY3D(cufinufft_plan *d_plan);
#endif
