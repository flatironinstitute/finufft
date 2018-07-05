#ifndef __UTILS_H__
#define __UTILS_H__

typedef double FLT;

// jfm timer class
#include <sys/time.h>
class CNTime {
  public:
    void start();
    double restart();
    double elapsedsec();
  private:
    struct timeval initial;
};
#endif
