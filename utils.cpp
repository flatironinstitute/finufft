#include "utils.h"
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

FLT relerrtwonorm(int n, CPX* a, CPX* b)
//||a-b||_2 / ||a||_2
{
  FLT err = 0.0, nrm = 0.0;
  for (int m=0; m<n; ++m) {
      nrm += real(conj(a[m])*a[m]);
      CPX diff = a[m]-b[m];
      err += real(conj(diff)*diff);
  }
  return sqrt(err)/nrm;
}
