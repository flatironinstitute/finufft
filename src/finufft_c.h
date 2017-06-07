#ifndef FINUFFT_C_H
#define FINUFFT_C_H
// header for FINUFFT interface to C, which works from both C and C++.
// utils.h must be previously included if in C++ mode (see finufft_c.h)

#include <complex.h>
//#include <complex>

// for when included into C, utils.h not available, so following needed.
// Cannot be typedefs to work with the _Complex type below; must be defines:
#ifndef __cplusplus
#ifdef SINGLE
  #define FLT float
#else
  #define FLT double
#endif
#endif

// this fails:
//  int finufft1d1_c(int nj,double* xj,std::complex<double>* cj,int iflag, double eps,int ms, std::complex<double>* fk);

// works but don't really understand it:
int finufft1d1_c(int nj,FLT* xj,FLT _Complex* cj,int iflag, FLT eps,int ms, FLT _Complex* fk);

#endif
