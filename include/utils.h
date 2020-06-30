// Header for utils.cpp, a little library of array and timer stuff.
// (rest of finufft defs and types are now in defs.h)

#ifndef UTILS_H
#define UTILS_H

#include "dataTypes.h"

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

// openmp helpers
int get_num_threads_parallel_block();

// thread-safe rand number generator for Windows platform
#ifdef _WIN32
#include <random>
int rand_r(unsigned int *seedp);
#endif

#endif  // UTILS_H
