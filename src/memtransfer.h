#ifndef __MEMTRANSFER_H__
#define __MEMTRANSFER_H__

#include <cufft.h>
#include "cufinufft.h"

int allocgpumem2d_plan(cufinufft_plan *d_plan);
int allocgpumem2d_nupts(cufinufft_plan *d_plan);

void freegpumemory2d(cufinufft_plan *d_plan);

#endif
