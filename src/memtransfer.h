#ifndef __MEMTRANSFER_H__
#define __MEMTRANSFER_H__

#include <cufft.h>

#include "finufft/utils.h"
#include "spread.h"

struct spread_opts;

struct cufinufft_plan {
  int M;
  int nf1;
  int nf2;
  int ms;
  int mt; 
  int fw_width;
  int iflag;

  int byte_now;
  FLT *fwkerhalf1;
  FLT *fwkerhalf2;
  FLT *h_fwkerhalf1;
  FLT *h_fwkerhalf2;

  FLT *kx;
  FLT *ky;
  CUCPX *c;
  CUCPX *fw;
  CUCPX *fk;

  FLT *h_kx;
  FLT *h_ky;
  CPX *h_c;
  CPX *h_fk;
  CPX *h_fw;

  FLT *kxsorted;
  FLT *kysorted;
  CUCPX *csorted;

  int *sortidx;
  int *binsize;
  int *binstartpts;
  int *numsubprob;
  int *subprob_to_bin;
  int *idxnupts;
  int *subprobstartpts;

  void *temp_storage;
  cufftHandle fftplan;
};

int allocgpumemory(spread_opts opts, cufinufft_plan *d_plan);
int copycpumem_to_gpumem(spread_opts opts, cufinufft_plan *d_plan);
int copygpumem_to_cpumem_fw(cufinufft_plan *d_plan);
int copygpumem_to_cpumem_fk(cufinufft_plan *d_mem);
int copygpumem_to_cpumem_c(cufinufft_plan *d_plan);
void free_gpumemory(spread_opts opts, cufinufft_plan *d_mem);

#endif
