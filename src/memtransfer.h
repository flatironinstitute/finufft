#ifndef __MEMTRANSFER_H__
#define __MEMTRANSFER_H__

#include <cufft.h>
#include "cufinufft.h"


int allocgpumemory(const cufinufft_opts opts, cufinufft_plan *d_plan);
void freegpumemory(const cufinufft_opts opts, cufinufft_plan *d_mem);

#endif
