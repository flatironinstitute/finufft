#include "utils.h"

extern "C" {
#include "finufft_c.h"
}

#include "finufft.h"

//#include <ccomplex>   // C++ complex which includes C-type complex

// wrappers for calling FINUFFT in C complex style, with default opts for now.
// The interface is the same as the C++ library but without the last opts arg.
// integer*4 for the array size arguments for now.
//
// The correct way to interface complex type from C to C++ is confusing, as
// is apparent by the comments in this file.
// Barnett 3/10/17

int finufft1d1_c(int nj,FLT* xj,FLT _Complex* cj,int iflag, FLT eps,int ms, FLT _Complex* fk)
//int finufft1d1_c(int nj,double* xj,dcomplex* cj,int iflag, double eps,int ms, dcomplex* fk)
{
  nufft_opts opts; finufft_default_opts(opts);
  return finufft1d1((INT)nj,xj,(CPX *)cj,iflag,eps,(INT)ms,(CPX *)fk,opts);
}

// todo: continue, once believe this is the best complex conversion...
