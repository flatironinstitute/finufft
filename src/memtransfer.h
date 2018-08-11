#ifndef __MEMTRANSFER_H__
#define __MEMTRANSFER_H__

#include <cufft.h>
#include "cufinufft.h"


int allocgpumemory(const cufinufft_opts opts, cufinufft_plan *d_plan);
int copycpumem_to_gpumem(const cufinufft_opts opts, cufinufft_plan *d_plan);
int copygpumem_to_cpumem_fw(cufinufft_plan *d_plan);
int copygpumem_to_cpumem_fk(cufinufft_plan *d_mem);
int copygpumem_to_cpumem_c(cufinufft_plan *d_plan);
void free_gpumemory(const cufinufft_opts opts, cufinufft_plan *d_mem);

#endif
