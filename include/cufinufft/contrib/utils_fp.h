// Header for utils_fp.cpp, a little library of low-level array stuff.
// These are functions which depend on single/double precision.
// (rest of finufft defs and types are now in defs.h)

#if (!defined(UTILS_FP_H) && !defined(CUFINUFFT_SINGLE)) || (!defined(UTILS_FPF_H) && defined(CUFINUFFT_SINGLE))
// Make sure we only include once per precision (as in finufft_eitherprec.h).
#ifndef CUFINUFFT_SINGLE
#define UTILS_FP_H
#else
#define UTILS_FPF_H
#endif


// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <cstdint>

#include <cufinufft_types.h>

// ahb's low-level array helpers
CUFINUFFT_FLT relerrtwonorm(CUFINUFFT_BIGINT n, CUFINUFFT_CPX* a, CUFINUFFT_CPX* b);
CUFINUFFT_FLT errtwonorm(CUFINUFFT_BIGINT n, CUFINUFFT_CPX* a, CUFINUFFT_CPX* b);
CUFINUFFT_FLT twonorm(CUFINUFFT_BIGINT n, CUFINUFFT_CPX* a);
CUFINUFFT_FLT infnorm(CUFINUFFT_BIGINT n, CUFINUFFT_CPX* a);
void arrayrange(CUFINUFFT_BIGINT n, CUFINUFFT_FLT* a, CUFINUFFT_FLT *lo, CUFINUFFT_FLT *hi);
void indexedarrayrange(CUFINUFFT_BIGINT n, CUFINUFFT_BIGINT* i, CUFINUFFT_FLT* a, CUFINUFFT_FLT *lo, CUFINUFFT_FLT *hi);
void arraywidcen(CUFINUFFT_BIGINT n, CUFINUFFT_FLT* a, CUFINUFFT_FLT *w, CUFINUFFT_FLT *c);

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
