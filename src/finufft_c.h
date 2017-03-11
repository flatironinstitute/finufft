#ifndef FINUFFT_C_H
#define FINUFFT_C_H

//#include <complex>
#include <complex.h>

#ifdef __cplusplus
  extern "C" {
#endif

  //  int finufft1d1_c(int nj,double* xj,std::complex<double>* cj,int iflag, double eps,int ms, std::complex<double>* fk);
  int finufft1d1_c(int nj,double* xj,double _Complex* cj,int iflag, double eps,int ms, double _Complex* fk);

#ifdef __cplusplus
}
#endif

#endif
