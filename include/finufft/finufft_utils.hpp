// Header for utils.cpp, a little library of low-level array stuff.
// These are just the functions which depend on single/double precision (FLT)

#pragma once

#include "finufft_core.h"
#include <cmath>
#include <finufft_common/common.h>

namespace finufft::utils {

template<typename T>
FINUFFT_EXPORT FINUFFT_ALWAYS_INLINE void FINUFFT_CDECL arrayrange(BIGINT n, const T *a,
                                                                   T *lo, T *hi)
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
FINUFFT_EXPORT FINUFFT_ALWAYS_INLINE void FINUFFT_CDECL arraywidcen(BIGINT n, const T *a,
                                                                    T *w, T *c)
// Writes out w = half-width and c = center of an interval enclosing all a[n]'s
// Only chooses a nonzero center if this increases w by less than fraction
// ARRAYWIDCEN_GROWFRAC defined in finufft_common/constants.h.
// This prevents rephasings which don't grow nf by much. 6/8/17
// If n==0, w and c are not finite.
{
  T lo, hi;
  arrayrange(n, a, &lo, &hi);
  *w = (hi - lo) / 2;
  *c = (hi + lo) / 2;
  if (std::abs(*c) < common::ARRAYWIDCEN_GROWFRAC * (*w)) {
    *w += std::abs(*c);
    *c = 0.0;
  }
}

// routines in finufft_utils.cpp ...
FINUFFT_EXPORT BIGINT next235even(BIGINT n);

// jfm's timer class
class FINUFFT_EXPORT CNTime {
public:
  FINUFFT_NEVER_INLINE void start();
  FINUFFT_NEVER_INLINE double restart();
  FINUFFT_NEVER_INLINE double elapsedsec() const;

private:
  double initial;
};

#ifdef _OPENMP
FINUFFT_NEVER_INLINE unsigned getOptimalThreadCount();
#endif

} // namespace finufft::utils

// thread-safe rand number generator for Windows platform
#ifdef _WIN32
#include <random>
namespace finufft {
namespace utils {
FINUFFT_EXPORT int FINUFFT_CDECL rand_r(unsigned int *seedp);
} // namespace utils
} // namespace finufft
#endif
