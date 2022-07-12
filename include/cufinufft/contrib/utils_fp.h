// Header for utils_fp.cpp, a little library of low-level array stuff.
// These are functions which depend on single/double precision.
// (rest of finufft defs and types are now in defs.h)

#if (!defined(UTILS_FP_H) && !defined(SINGLE)) || (!defined(UTILS_FPF_H) && defined(SINGLE))
// Make sure we only include once per precision (as in finufft_eitherprec.h).
#ifndef SINGLE
#define UTILS_FP_H
#else
#define UTILS_FPF_H
#endif


// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <stdint.h>

#include <complex>          // C++ type complex
#include <cuComplex.h>
#include "dataTypes.h"


#undef EPSILON
#undef IMA
#undef FABS
#undef CUCPX
#undef CUFFT_TYPE
#undef CUFFT_EX
#undef SET_NF_TYPE12

// Compile-flag choice of single or double (default) precision:
// (Note in the other codes, CUFINUFFT_FLT is "double" or "float", CPX same but complex)
#ifdef SINGLE
  // machine epsilon for rounding
  #define EPSILON (float)6e-08
  #define IMA complex<float>(0.0,1.0)
  #define FABS(x) fabs(x)
  #define CUCPX cuFloatComplex
  #define CUFFT_TYPE CUFFT_C2C
  #define CUFFT_EX cufftExecC2C
  #define SET_NF_TYPE12 set_nf_type12f
#else
  // machine epsilon for rounding
  #define EPSILON (double)1.1e-16
  #define IMA complex<double>(0.0,1.0)
  #define FABS(x) fabsf(x)
  #define CUCPX cuDoubleComplex
  #define CUFFT_TYPE CUFFT_Z2Z
  #define CUFFT_EX cufftExecZ2Z
  #define SET_NF_TYPE12 set_nf_type12
#endif


// ahb's low-level array helpers
CUFINUFFT_FLT relerrtwonorm(BIGINT n, CPX* a, CPX* b);
CUFINUFFT_FLT errtwonorm(BIGINT n, CPX* a, CPX* b);
CUFINUFFT_FLT twonorm(BIGINT n, CPX* a);
CUFINUFFT_FLT infnorm(BIGINT n, CPX* a);
void arrayrange(BIGINT n, CUFINUFFT_FLT* a, CUFINUFFT_FLT *lo, CUFINUFFT_FLT *hi);
void indexedarrayrange(BIGINT n, BIGINT* i, CUFINUFFT_FLT* a, CUFINUFFT_FLT *lo, CUFINUFFT_FLT *hi);
void arraywidcen(BIGINT n, CUFINUFFT_FLT* a, CUFINUFFT_FLT *w, CUFINUFFT_FLT *c);

// Random numbers: crappy unif random number generator in [0,1):
//#define rand01() (((CUFINUFFT_FLT)(rand()%RAND_MAX))/RAND_MAX)
#define rand01() ((CUFINUFFT_FLT)rand()/RAND_MAX)
// unif[-1,1]:
#define randm11() (2*rand01() - (CUFINUFFT_FLT)1.0)
// complex unif[-1,1] for Re and Im:
#define crandm11() (randm11() + IMA*randm11())

// Thread-safe seed-carrying versions of above (x is ptr to seed)...
#define rand01r(x) ((CUFINUFFT_FLT)rand_r(x)/RAND_MAX)
// unif[-1,1]:
#define randm11r(x) (2*rand01r(x) - (CUFINUFFT_FLT)1.0)
// complex unif[-1,1] for Re and Im:
#define crandm11r(x) (randm11r(x) + IMA*randm11r(x))


#endif  // UTILS_FP_H
