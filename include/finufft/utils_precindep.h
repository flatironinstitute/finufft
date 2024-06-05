// Header for utils_precindep.cpp, a little library of array and timer stuff.
// Only the precision-independent routines here (get compiled once)

#ifndef UTILS_PRECINDEP_H
#define UTILS_PRECINDEP_H

#include "defs.h"
// for CNTime...
// using chrono since the interface is portable between linux and windows
#include <chrono>

namespace finufft {
namespace utils {

FINUFFT_EXPORT BIGINT FINUFFT_CDECL next235even(BIGINT n);

// jfm's timer class
class FINUFFT_EXPORT CNTime {
public:
  void start();
  double restart();
  double elapsedsec();

private:
  double initial;
};

// openmp helpers
int get_num_threads_parallel_block();

} // namespace utils
} // namespace finufft

// thread-safe rand number generator for Windows platform
#ifdef _WIN32
#include <random>
namespace finufft {
namespace utils {
FINUFFT_EXPORT int FINUFFT_CDECL rand_r(unsigned int *seedp);
} // namespace utils
} // namespace finufft
#endif

#endif // UTILS_PRECINDEP_H
