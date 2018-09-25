// A little library of low-level array manipulations and timers.
// For its embryonic self-test see ../test/testutils.cpp, which only tests
// the next235 for now.

#include "utils.h"

// ------------ complex array utils

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
{
  FLT lo,hi;
  arrayrange(n,a,&lo,&hi);
  *w = (hi-lo)/2;
  *c = (hi+lo)/2;
  if (FABS(*c)<ARRAYWIDCEN_GROWFRAC*(*w)) {
    *w += FABS(*c);
    *c = 0.0;
  }
}

BIGINT next235even(BIGINT n)
// finds even integer not less than n, with prime factors no larger than 5
// (ie, "smooth"). Adapted from fortran in hellskitchen.  Barnett 2/9/17
// changed INT64 type 3/28/17. Runtime is around n*1e-11 sec for big n.
{
  if (n<=2) return 2;
  if (n%2 == 1) n+=1;   // even
  BIGINT nplus = n-2;   // to cancel out the +=2 at start of loop
  BIGINT numdiv = 2;    // a dummy that is >1
  while (numdiv>1) {
    nplus += 2;         // stays even
    numdiv = nplus;
    while (numdiv%2 == 0) numdiv /= 2;  // remove all factors of 2,3,5...
    while (numdiv%3 == 0) numdiv /= 3;
    while (numdiv%5 == 0) numdiv /= 5;
  }
  return nplus;
}

// ----------------------- helpers for timing (always stay double prec)...
using namespace std;

void CNTime::start()
{
  gettimeofday(&initial, 0);
}

double CNTime::restart()
// Barnett changed to returning in sec
{
  double delta = this->elapsedsec();
  this->start();
  return delta;
}

double CNTime::elapsedsec()
// returns answers as double, in seconds, to microsec accuracy. Barnett 5/22/18
{
  struct timeval now;
  gettimeofday(&now, 0);
  double nowsec = (double)now.tv_sec + 1e-6*now.tv_usec;
  double initialsec = (double)initial.tv_sec + 1e-6*initial.tv_usec;
  return nowsec - initialsec;
}
