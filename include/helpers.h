#ifndef HELPERS_H
#define HELPERS_H

#include <dataTypes.h>

BIGINT next235even(BIGINT n);


// jfm's timer class
#include <sys/time.h>
class CNTime {
 public:
  void start();
  double restart();
  double elapsedsec();
 private:
  struct timeval initial;
};

int ndims_from_Ns(BIGINT N1, BIGINT N2, BIGINT N3);

#endif
