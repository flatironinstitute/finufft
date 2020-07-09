#include "utils.h"
#include "utils_fp.h"


// ------------ complex array utils ---------------------------------

FLT relerrtwonorm(BIGINT n, CPX* a, CPX* b)
// ||a-b||_2 / ||a||_2
{
  FLT err = 0.0, nrm = 0.0;
  for (BIGINT m=0; m<n; ++m) {
    nrm += real(conj(a[m])*a[m]);
    CPX diff = a[m]-b[m];
    err += real(conj(diff)*diff);
  }
  return sqrt(err/nrm);
}

FLT errtwonorm(BIGINT n, CPX* a, CPX* b)
// ||a-b||_2
{
  FLT err = 0.0;   // compute error 2-norm
  for (BIGINT m=0; m<n; ++m) {
    CPX diff = a[m]-b[m];
    err += real(conj(diff)*diff);
  }
  return sqrt(err);
}

FLT twonorm(BIGINT n, CPX* a)
// ||a||_2
{
  FLT nrm = 0.0;
  for (BIGINT m=0; m<n; ++m)
    nrm += real(conj(a[m])*a[m]);
  return sqrt(nrm);
}

FLT infnorm(BIGINT n, CPX* a)
// ||a||_infty
{
  FLT nrm = 0.0;
  for (BIGINT m=0; m<n; ++m) {
    FLT aa = real(conj(a[m])*a[m]);
    if (aa>nrm) nrm = aa;
  }
  return sqrt(nrm);
}

void arrayrange(BIGINT n, FLT* a, FLT *lo, FLT *hi)
// With a a length-n array, writes out min(a) to lo and max(a) to hi,
// so that all a values lie in [lo,hi].
// If n==0, lo and hi are not finite.
{
  *lo = INFINITY; *hi = -INFINITY;
  for (BIGINT m=0; m<n; ++m) {
    if (a[m]<*lo) *lo = a[m];
    if (a[m]>*hi) *hi = a[m];
  }
}

void indexedarrayrange(BIGINT n, BIGINT* i, FLT* a, FLT *lo, FLT *hi)
// With i a list of n indices, and a an array of length max(i), writes out
// min(a(i)) to lo and max(a(i)) to hi, so that all a(i) values lie in [lo,hi].
// This is not currently used in FINUFFT v1.2.
{
  *lo = INFINITY; *hi = -INFINITY;
  for (BIGINT m=0; m<n; ++m) {
    FLT A=a[i[m]];
    if (A<*lo) *lo = A;
    if (A>*hi) *hi = A;
  }
}

void arraywidcen(BIGINT n, FLT* a, FLT *w, FLT *c)
// Writes out w = half-width and c = center of an interval enclosing all a[n]'s
// Only chooses a nonzero center if this increases w by less than fraction
// ARRAYWIDCEN_GROWFRAC defined in defs.h.
// This prevents rephasings which don't grow nf by much. 6/8/17
// If n==0, w and c are not finite.
{
  FLT lo,hi;
  arrayrange(n,a,&lo,&hi);
  *w = (hi-lo)/2;
  *c = (hi+lo)/2;
  if (std::abs(*c)<ARRAYWIDCEN_GROWFRAC*(*w)) {
    *w += std::abs(*c);
    *c = 0.0;
  }
}
