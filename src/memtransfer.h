#ifndef __MEMTRANSFER_H__
#define __MEMTRANSFER_H__

#include <cufft.h>
#include "cufinufft.h"

int allocgpumemory1d(const cufinufft_opts opts, cufinufft_plan *d_plan);
void freegpumemory1d(const cufinufft_opts opts, cufinufft_plan *d_mem);

int allocgpumemory2d(const cufinufft_opts opts, cufinufft_plan *d_plan);
void freegpumemory2d(const cufinufft_opts opts, cufinufft_plan *d_mem);

int allocgpumemory3d(const cufinufft_opts opts, cufinufft_plan *d_plan);
void freegpumemory3d(const cufinufft_opts opts, cufinufft_plan *d_mem);
#endif
