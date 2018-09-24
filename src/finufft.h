// minimum definitions needed for interface to FINUFFT library, from C++ or C

#ifndef FINUFFT_H
#define FINUFFT_H

// ----------------- data type definitions ----------------------------------
// (note: non-interface precision- and omp-dependent defs are in defs.h)

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <stdint.h>

// All indexing in library that potentially can exceed 2^31 uses 64-bit signed.
// This includes all calling arguments (eg M,N) that could be huge someday...
typedef int64_t BIGINT;

// decide which kind of complex numbers to use in interface...
#ifdef __cplusplus
#include <complex>          // C++ type
#else
#include <complex.h>        // C99 type
#endif

// Precision-independent real and complex types for interfacing...
#ifdef SINGLE
  typedef float FLT;
  #ifdef __cplusplus
    typedef std::complex<float> CPX;
  #else
    typedef float complex CPX;
  #endif
#else
  typedef double FLT;
  #ifdef __cplusplus
    typedef std::complex<double> CPX;
  #else
    typedef double complex CPX;
  #endif
#endif


// ------------------- the user input options struct ------------------------
typedef struct {      // Note: defaults in common/finufft_default_opts()
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
} nufft_opts;


// ------------------ library provides ------------------------------------
#ifdef __cplusplus
extern "C"
{
#endif
void finufft_default_opts(nufft_opts *o);
int finufft1d1(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
	       CPX* fk, nufft_opts opts);
int finufft1d2(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
	       CPX* fk, nufft_opts opts);
int finufft1d3(BIGINT nj,FLT* x,CPX* c,int iflag,FLT eps,BIGINT nk, FLT* s, CPX* f, nufft_opts opts);

int finufft2d1(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);
int finufft2d1many(int ndata, BIGINT nj, FLT* xj, FLT *yj, CPX* c, int iflag,
                   FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);
int finufft2d2(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);
int finufft2d2many(int ndata, BIGINT nj, FLT* xj, FLT *yj, CPX* c, int iflag,
                   FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);
int finufft2d3(BIGINT nj,FLT* x,FLT *y,CPX* cj,int iflag,FLT eps,BIGINT nk, FLT* s, FLT* t, CPX* fk, nufft_opts opts);

int finufft3d1(BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk, nufft_opts opts);
int finufft3d2(BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk, nufft_opts opts);
int finufft3d3(BIGINT nj,FLT* x,FLT *y,FLT *z, CPX* cj,int iflag,
	       FLT eps,BIGINT nk,FLT* s, FLT* t, FLT *u,
	       CPX* fk, nufft_opts opts);
#ifdef __cplusplus
}
#endif


#endif   // FINUFFT_H
