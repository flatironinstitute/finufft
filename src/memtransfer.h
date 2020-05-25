#ifndef __MEMTRANSFER_H__
#define __MEMTRANSFER_H__

#include <cufinufft.h>

int allocgpumem1d_plan(cufinufft_plan *d_plan);
int allocgpumem1d_nupts(cufinufft_plan *d_plan);
void freegpumemory1d(cufinufft_plan *d_plan);

int allocgpumem2d_plan(cufinufft_plan *d_plan);
int allocgpumem2d_nupts(cufinufft_plan *d_plan);
void freegpumemory2d(cufinufft_plan *d_plan);

int allocgpumem3d_plan(cufinufft_plan *d_plan);
int allocgpumem3d_nupts(cufinufft_plan *d_plan);
void freegpumemory3d(cufinufft_plan *d_plan);
#endif
