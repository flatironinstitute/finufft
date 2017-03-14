#include "utils.h"

// ------------ complex array utils

double relerrtwonorm(BIGINT n, dcomplex* a, dcomplex* b)
// ||a-b||_2 / ||a||_2
{
  double err = 0.0, nrm = 0.0;
  for (BIGINT m=0; m<n; ++m) {
    nrm += real(conj(a[m])*a[m]);
    dcomplex diff = a[m]-b[m];
    err += real(conj(diff)*diff);
  }
  return sqrt(err/nrm);
}
double errtwonorm(BIGINT n, dcomplex* a, dcomplex* b)
// ||a-b||_2
{
  double err = 0.0;   // compute error 2-norm
  for (BIGINT m=0; m<n; ++m) {
    dcomplex diff = a[m]-b[m];
    err += real(conj(diff)*diff);
  }
  return sqrt(err);
}
double twonorm(BIGINT n, dcomplex* a)
// ||a||_2
{
  double nrm = 0.0;
  for (BIGINT m=0; m<n; ++m)
    nrm += real(conj(a[m])*a[m]);
  return sqrt(nrm);
}
double infnorm(BIGINT n, dcomplex* a)
// ||a||_infty
{
  double nrm = 0.0;
  for (BIGINT m=0; m<n; ++m) {
    double aa = real(conj(a[m])*a[m]);
    if (aa>nrm) nrm = aa;
  }
  return sqrt(nrm);
}

void arrayrange(BIGINT n, double* a, double *lo, double *hi)
// writes out bounds on values in array to lo and hi, so all a in [lo,hi]
{
  *lo = INFINITY; *hi = -INFINITY;
  for (BIGINT m=0; m<n; ++m) {
    if (a[m]<*lo) *lo = a[m];
    if (a[m]>*hi) *hi = a[m];
  }
}

void arraywidcen(BIGINT n, double* a, double *w, double *c)
// writes out w = half-width and c = center of interval enclosing all a[n]'s
{
  double lo,hi;
  arrayrange(n,a,&lo,&hi);
  *w = (hi-lo)/2;
  *c = (hi+lo)/2;
}

BIGINT next235even(BIGINT n)
// finds even integer not less than n, with prime factors no larger than 5
// (ie, "smooth"). Adapted from fortran in hellskitchen.  Barnett 2/9/17
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


// ----------------------- helpers for timing...
using namespace std;

void CNTime::start()
{
  gettimeofday(&initial, 0);
}

int CNTime::restart()
{
  int delta = this->elapsed();
  this->start();
  return delta;
}

int CNTime::elapsed()
//  returns answers as integer number of milliseconds
{
  struct timeval now;
  gettimeofday(&now, 0);
  int delta = 1000 * (now.tv_sec - (initial.tv_sec + 1));
  delta += (now.tv_usec + (1000000 - initial.tv_usec)) / 1000;
  return delta;
}

double CNTime::elapsedsec()
//  returns answers as double in sec
{
  return (double)(this->elapsed()/1e3);
}
