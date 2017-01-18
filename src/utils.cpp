#include "utils.h"

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
