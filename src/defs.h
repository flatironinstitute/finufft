#ifndef DEFS_H
#define DEFS_H

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <stdint.h>

#include <complex>          // C++ type complex
#include <fftw3.h>          // needed so can typedef FFTW_CPX

typedef std::complex<double> dcomplex;  // slightly sneaky since duplicated by mwrap

// Compile-flag choice of single or double (default) precision:
// (Note in the other codes, FLT is "double" or "float", CPX same but complex)
#ifdef SINGLE
  // machine epsilon for rounding
  #define EPSILON (float)6e-08
  typedef float FLT;
  typedef std::complex<float> CPX;
  #define IMA std::complex<float>(0.0,1.0)
  #define FABS(x) fabs(x)
  typedef fftwf_complex FFTW_CPX;           //  single-prec has fftwf_*
  typedef fftwf_plan FFTW_PLAN;
  #define FFTW_INIT fftwf_init_threads
  #define FFTW_PLAN_TH fftwf_plan_with_nthreads
  #define FFTW_ALLOC_RE fftwf_alloc_real
  #define FFTW_ALLOC_CPX fftwf_alloc_complex
  #define FFTW_PLAN_1D fftwf_plan_dft_1d
  #define FFTW_PLAN_2D fftwf_plan_dft_2d
  #define FFTW_PLAN_3D fftwf_plan_dft_3d
  #define FFTW_PLAN_MANY_DFT fftwf_plan_many_dft
  #define FFTW_EX fftwf_execute
  #define FFTW_DE fftwf_destroy_plan
  #define FFTW_FR fftwf_free
  #define FFTW_FORGET_WISDOM fftwf_forget_wisdom
#else
  // machine epsilon for rounding
  #define EPSILON (double)1.1e-16
  typedef double FLT;
  typedef std::complex<double> CPX;
  #define IMA std::complex<double>(0.0,1.0)
  #define FABS(x) fabsf(x)
  typedef fftw_complex FFTW_CPX;           // double-prec has fftw_*
  typedef fftw_plan FFTW_PLAN;
  #define FFTW_INIT fftw_init_threads
  #define FFTW_PLAN_TH fftw_plan_with_nthreads
  #define FFTW_ALLOC_RE fftw_alloc_real
  #define FFTW_ALLOC_CPX fftw_alloc_complex
  #define FFTW_PLAN_1D fftw_plan_dft_1d
  #define FFTW_PLAN_2D fftw_plan_dft_2d
  #define FFTW_PLAN_3D fftw_plan_dft_3d
  #define FFTW_PLAN_MANY_DFT fftw_plan_many_dft
  #define FFTW_EX fftw_execute
  #define FFTW_DE fftw_destroy_plan
  #define FFTW_FR fftw_free
  #define FFTW_FORGET_WISDOM fftw_forget_wisdom
#endif

// All indexing in library that potentially can exceed 2^31 uses 64-bit signed.
// This includes all calling arguments (eg M,N) that could be huge someday...
typedef int64_t BIGINT;

// Global error codes for the library...
#define ERR_EPS_TOO_SMALL        1
#define ERR_MAXNALLOC            2
#define ERR_SPREAD_BOX_SMALL     3
#define ERR_SPREAD_PTS_OUT_RANGE 4
#define ERR_SPREAD_ALLOC         5
#define ERR_SPREAD_DIR           6
#define ERR_UPSAMPFAC_TOO_SMALL  7
#define HORNER_WRONG_BETA        8
#define ERR_NDATA_NOTVALID       9

#endif
