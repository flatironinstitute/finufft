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
