// Header for utils.cpp, a little library of low-level array stuff.
// These are just the functions which depend on single/double precision (FLT)

#ifndef UTILS_H
#define UTILS_H

#include "finufft/defs.h"

namespace finufft {
namespace utils {

// ahb's low-level array helpers
FINUFFT_EXPORT FLT FINUFFT_CDECL relerrtwonorm(BIGINT n, CPX *a, CPX *b);
FINUFFT_EXPORT FLT FINUFFT_CDECL errtwonorm(BIGINT n, CPX *a, CPX *b);
FINUFFT_EXPORT FLT FINUFFT_CDECL twonorm(BIGINT n, CPX *a);
FINUFFT_EXPORT FLT FINUFFT_CDECL infnorm(BIGINT n, CPX *a);
FINUFFT_EXPORT void FINUFFT_CDECL arrayrange(BIGINT n, FLT *a, FLT *lo, FLT *hi);
FINUFFT_EXPORT void FINUFFT_CDECL indexedarrayrange(BIGINT n, BIGINT *i, FLT *a, FLT *lo,
                                                    FLT *hi);
FINUFFT_EXPORT void FINUFFT_CDECL arraywidcen(BIGINT n, FLT *a, FLT *w, FLT *c);

} // namespace utils
} // namespace finufft

#endif // UTILS_H
