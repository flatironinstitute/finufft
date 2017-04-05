#include "utils.h"
extern "C" {
#include "finufft_c.h"
}
#include "finufft.h"

//#include <ccomplex>   // C++ complex which includes C-type complex

// wrappers for calling FINUFFT in C complex style, with default opts.
// The interface is the same as the C++ library but without the last arg.
// integer*4 for the array size arguments for now.
// Barnett 3/10/17

int finufft1d1_c(int nj,FLT* xj,FLT _Complex* cj,int iflag, FLT eps,int ms, FLT _Complex* fk)
//int finufft1d1_c(int nj,double* xj,dcomplex* cj,int iflag, double eps,int ms, dcomplex* fk)
{
  nufft_opts opts;
    printf("eps=%.3g\n",eps);

  return finufft1d1((INT)nj,xj,(CPX *)cj,iflag,eps,(INT)ms,(CPX *)fk,opts);
}
