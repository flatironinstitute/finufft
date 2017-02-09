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

void arrayrange(BIGINT n, double* a, double& lo, double &hi)
// writes out bounds on values in array to lo and hi, so all a in [lo,hi]
{
  lo = INFINITY; hi = -INFINITY;
  for (BIGINT m=0; m<n; ++m) {
    if (a[m]<lo) lo = a[m];
    if (a[m]>hi) hi = a[m];
  }
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
