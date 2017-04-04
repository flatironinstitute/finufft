#ifndef FINUFFT_C_H
#define FINUFFT_C_H

//#include <complex>
#include <complex.h>
#include "utils.h"

#ifdef __cplusplus
  extern "C" {
#endif

  //  int finufft1d1_c(int nj,double* xj,std::complex<double>* cj,int iflag, double eps,int ms, std::complex<double>* fk);
  int finufft1d1_c(int nj,FLT* xj,FLT _Complex* cj,int iflag, FLT eps,int ms, FLT _Complex* fk);

#ifdef __cplusplus
}
#endif

#endif
