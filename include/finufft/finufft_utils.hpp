// Header for utils.cpp, a little library of low-level array stuff.
// These are just the functions which depend on single/double precision (FLT)

#pragma once

#include <chrono>
#include <cmath>

#include "finufft_core.h"

//  for CNTime...
//  using chrono since the interface is portable between linux and windows

namespace finufft::utils {

template<typename T>
FINUFFT_ALWAYS_INLINE void arrayrange(BIGINT n, const T *a, T *lo, T *hi)
// With a a length-n array, writes out min(a) to lo and max(a) to hi,
// so that all a values lie in [lo,hi].
// If n==0, lo and hi are not finite.
{
  *lo = INFINITY;
  *hi = -INFINITY;
  for (BIGINT m = 0; m < n; ++m) {
    if (a[m] < *lo) *lo = a[m];
    if (a[m] > *hi) *hi = a[m];
  }
}
template<typename T>
FINUFFT_ALWAYS_INLINE void arraywidcen(BIGINT n, const T *a, T *w, T *c)
// Writes out w = half-width and c = center of an interval enclosing all a[n]'s
// Only chooses a nonzero center if this increases w by less than fraction
// ARRAYWIDCEN_GROWFRAC defined in finufft_core.h.
// This prevents rephasings which don't grow nf by much. 6/8/17
// If n==0, w and c are not finite.
{
  T lo, hi;
  arrayrange(n, a, &lo, &hi);
  *w = (hi - lo) / 2;
  *c = (hi + lo) / 2;
  if (std::abs(*c) < ARRAYWIDCEN_GROWFRAC * (*w)) {
    *w += std::abs(*c);
    *c = 0.0;
  }
}

BIGINT next235even(BIGINT n);

// jfm's timer class
class FINUFFT_EXPORT CNTime {
public:
  FINUFFT_NEVER_INLINE void start();
  FINUFFT_NEVER_INLINE double restart();
  FINUFFT_NEVER_INLINE double elapsedsec() const;

private:
  double initial;
};

// openmp helpers
int get_num_threads_parallel_block();

} // namespace finufft::utils

// thread-safe rand number generator for Windows platform
#ifdef _WIN32
#include <random>
namespace finufft {
namespace utils {
int rand_r(unsigned int *seedp);
} // namespace utils
} // namespace finufft
#endif
