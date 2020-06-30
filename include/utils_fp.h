// Header for utils_fp.cpp, a little library of low-level array stuff.
// These are functions which depend on single/double precision.
// (rest of finufft defs and types are now in defs.h)

#ifndef UTILS_FP_H
#define UTILS_FP_H

#include "dataTypes.h"

// ahb's low-level array helpers
FLT relerrtwonorm(BIGINT n, CPX* a, CPX* b);
FLT errtwonorm(BIGINT n, CPX* a, CPX* b);
FLT twonorm(BIGINT n, CPX* a);
FLT infnorm(BIGINT n, CPX* a);
void arrayrange(BIGINT n, FLT* a, FLT *lo, FLT *hi);
void indexedarrayrange(BIGINT n, BIGINT* i, FLT* a, FLT *lo, FLT *hi);
void arraywidcen(BIGINT n, FLT* a, FLT *w, FLT *c);

#endif  // UTILS_FP_H
