// C-style interface to FINUFFT library that is used for MWRAP interface.
// Barnett 3/21/17

#include "../src/finufft.h"

int finufft1d1_m(int nj,double* xj,dcomplex* cj,int iflag,double eps,int ms,
		 dcomplex* fk,int debug,int nthreads)
{
  nufft_opts opts;
  opts.debug = debug;
  if (nthreads>0) MY_OMP_SET_NUM_THREADS(nthreads);
  return finufft1d1((BIGINT)nj,xj,cj,iflag,eps,(BIGINT)ms,fk,opts);
}
