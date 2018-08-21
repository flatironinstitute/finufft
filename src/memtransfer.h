#ifndef __MEMTRANSFER_H__
#define __MEMTRANSFER_H__

#include <cufft.h>
#include "cufinufft.h"


int allocgpumemory(const cufinufft_opts opts, cufinufft_plan *d_plan);
void free_gpumemory(const cufinufft_opts opts, cufinufft_plan *d_mem);

#endif
