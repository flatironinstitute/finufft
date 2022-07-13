#include "cufinufft/contrib/utils_fp.h"
#include "cufinufft/contrib/utils.h"

// ------------ complex array utils ---------------------------------

CUFINUFFT_FLT relerrtwonorm(CUFINUFFT_BIGINT n, CUFINUFFT_CPX *a, CUFINUFFT_CPX *b)
// ||a-b||_2 / ||a||_2
{
    CUFINUFFT_FLT err = 0.0, nrm = 0.0;
    for (CUFINUFFT_BIGINT m = 0; m < n; ++m) {
        nrm += real(conj(a[m]) * a[m]);
        CUFINUFFT_CPX diff = a[m] - b[m];
        err += real(conj(diff) * diff);
    }
    return sqrt(err / nrm);
}

CUFINUFFT_FLT errtwonorm(CUFINUFFT_BIGINT n, CUFINUFFT_CPX *a, CUFINUFFT_CPX *b)
// ||a-b||_2
{
    CUFINUFFT_FLT err = 0.0; // compute error 2-norm
    for (CUFINUFFT_BIGINT m = 0; m < n; ++m) {
        CUFINUFFT_CPX diff = a[m] - b[m];
        err += real(conj(diff) * diff);
    }
    return sqrt(err);
}

CUFINUFFT_FLT twonorm(CUFINUFFT_BIGINT n, CUFINUFFT_CPX *a)
// ||a||_2
{
    CUFINUFFT_FLT nrm = 0.0;
    for (CUFINUFFT_BIGINT m = 0; m < n; ++m)
        nrm += real(conj(a[m]) * a[m]);
    return sqrt(nrm);
}

CUFINUFFT_FLT infnorm(CUFINUFFT_BIGINT n, CUFINUFFT_CPX *a)
// ||a||_infty
{
    CUFINUFFT_FLT nrm = 0.0;
    for (CUFINUFFT_BIGINT m = 0; m < n; ++m) {
        CUFINUFFT_FLT aa = real(conj(a[m]) * a[m]);
        if (aa > nrm)
            nrm = aa;
    }
    return sqrt(nrm);
}

void arrayrange(CUFINUFFT_BIGINT n, CUFINUFFT_FLT *a, CUFINUFFT_FLT *lo, CUFINUFFT_FLT *hi)
// With a a length-n array, writes out min(a) to lo and max(a) to hi,
// so that all a values lie in [lo,hi].
// If n==0, lo and hi are not finite.
{
    *lo = INFINITY;
    *hi = -INFINITY;
    for (CUFINUFFT_BIGINT m = 0; m < n; ++m) {
        if (a[m] < *lo)
            *lo = a[m];
        if (a[m] > *hi)
            *hi = a[m];
    }
}

void indexedarrayrange(CUFINUFFT_BIGINT n, CUFINUFFT_BIGINT *i, CUFINUFFT_FLT *a, CUFINUFFT_FLT *lo, CUFINUFFT_FLT *hi)
// With i a list of n indices, and a an array of length max(i), writes out
// min(a(i)) to lo and max(a(i)) to hi, so that all a(i) values lie in [lo,hi].
// This is not currently used in FINUFFT v1.2.
{
    *lo = INFINITY;
    *hi = -INFINITY;
    for (CUFINUFFT_BIGINT m = 0; m < n; ++m) {
        CUFINUFFT_FLT A = a[i[m]];
        if (A < *lo)
            *lo = A;
        if (A > *hi)
            *hi = A;
    }
}

void arraywidcen(CUFINUFFT_BIGINT n, CUFINUFFT_FLT *a, CUFINUFFT_FLT *w, CUFINUFFT_FLT *c)
// Writes out w = half-width and c = center of an interval enclosing all a[n]'s
// Only chooses a nonzero center if this increases w by less than fraction
// ARRAYWIDCEN_GROWFRAC defined in defs.h.
// This prevents rephasings which don't grow nf by much. 6/8/17
// If n==0, w and c are not finite.
{
    CUFINUFFT_FLT lo, hi;
    arrayrange(n, a, &lo, &hi);
    *w = (hi - lo) / 2;
    *c = (hi + lo) / 2;
    if (std::abs(*c) < ARRAYWIDCEN_GROWFRAC * (*w)) {
        *w += std::abs(*c);
        *c = 0.0;
    }
}
