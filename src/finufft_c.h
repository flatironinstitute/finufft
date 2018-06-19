#ifndef FINUFFT_C_H
#define FINUFFT_C_H
// header for FINUFFT interface to C, which works from both C and C++.
// utils.h must be previously included if in C++ mode (see finufft_c.cpp)

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

typedef struct nufft_c_opts {
  int debug;          // 0: silent, 1: text basic timing output
  int spread_debug;   // passed to spread_opts, 0 (no text) 1 (some) or 2 (lots)
  int spread_sort;    // passed to spread_opts, 0 (don't sort) 1 (do) or 2 (heuristic)
  int spread_kerevalmeth; // "     spread_opts, 0: exp(sqrt()), 1: Horner ppval (faster)
  int spread_kerpad;  // passed to spread_opts, 0: don't pad to mult of 4, 1: do
  int chkbnds;        // 0: don't check if input NU pts in [-3pi,3pi], 1: do
  int fftw;           // 0:FFTW_ESTIMATE, or 1:FFTW_MEASURE (slow plan but faster)
  int modeord;        // 0: CMCL-style increasing mode ordering (neg to pos), or
                      // 1: FFT-style mode ordering (affects type-1,2 only)
  FLT upsampfac;      // upsampling ratio sigma, either 2.0 (standard) or 1.25 (small FFT)
} nufft_c_opts;

void finufft_default_c_opts(nufft_c_opts *o);

// this fails:
//  int finufft1d1_c(int nj,double* xj,std::complex<double>* cj,int iflag, double eps,int ms, std::complex<double>* fk);

// interface definitions. They work but I don't really understand _Complex :

int finufft1d1_c(int nj,FLT* xj,FLT _Complex* cj,int iflag, FLT eps,int ms, FLT _Complex* fk, nufft_c_opts copts);
int finufft1d2_c(int nj,FLT* xj,FLT _Complex* cj,int iflag, FLT eps,int ms, FLT _Complex* fk, nufft_c_opts copts);
int finufft1d3_c(int j,FLT* x,FLT _Complex* c,int iflag,FLT eps,int nk, FLT* s, FLT _Complex* f, nufft_c_opts copts);
int finufft2d1_c(int nj,FLT* xj,FLT *yj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt,FLT _Complex* fk, nufft_c_opts copts);
int finufft2d2_c(int nj,FLT* xj,FLT *yj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt, FLT _Complex* fk, nufft_c_opts copts);
int finufft2d3_c(int nj,FLT* x,FLT *y,FLT _Complex* c,int iflag,FLT eps,int nk, FLT* s, FLT *t,FLT _Complex* f, nufft_c_opts copts);
int finufft3d1_c(int nj,FLT* xj,FLT* yj,FLT *zj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt, int mu,FLT _Complex* fk, nufft_c_opts copts);
int finufft3d2_c(int nj,FLT* xj,FLT *yj,FLT *zj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt, int mu, FLT _Complex* fk, nufft_c_opts copts);
int finufft3d3_c(int nj,FLT* x,FLT *y,FLT *z,FLT _Complex* c,int iflag,FLT eps,int nk, FLT* s, FLT *t,FLT *u,FLT _Complex* f, nufft_c_opts copts);


#endif
