// Header for utils.cpp, a little library of low-level array stuff.
// These are just the functions which depend on single/double precision (FLT)

#ifndef UTILS_H
#define UTILS_H

#include "finufft/finufft_core.h"
//  for CNTime...
//  using chrono since the interface is portable between linux and windows
#include <chrono>

namespace finufft {
namespace utils {

// ahb's low-level array helpers
template<typename T>
FINUFFT_EXPORT T FINUFFT_CDECL relerrtwonorm(BIGINT n, std::complex<T> *a,
                                             std::complex<T> *b)
// ||a-b||_2 / ||a||_2
{
  T err = 0.0, nrm = 0.0;
  for (BIGINT m = 0; m < n; ++m) {
    nrm += std::norm(a[m]);
    err += std::norm(a[m] - b[m]);
  }
  return sqrt(err / nrm);
}
template<typename T>
FINUFFT_EXPORT T FINUFFT_CDECL errtwonorm(BIGINT n, std::complex<T> *a,
                                          std::complex<T> *b)
// ||a-b||_2
{
  T err = 0.0; // compute error 2-norm
  for (BIGINT m = 0; m < n; ++m) err += std::norm(a[m] - b[m]);
  return sqrt(err);
}
template<typename T>
FINUFFT_EXPORT T FINUFFT_CDECL twonorm(BIGINT n, std::complex<T> *a)
// ||a||_2
{
  T nrm = 0.0;
  for (BIGINT m = 0; m < n; ++m) nrm += std::norm(a[m]);
  return sqrt(nrm);
}
template<typename T>
FINUFFT_EXPORT T FINUFFT_CDECL infnorm(BIGINT n, std::complex<T> *a)
// ||a||_infty
{
  T nrm = 0.0;
  for (BIGINT m = 0; m < n; ++m) nrm = std::max(nrm, std::norm(a[m]));
  return sqrt(nrm);
}
template<typename T>
FINUFFT_EXPORT void FINUFFT_CDECL arrayrange(BIGINT n, T *a, T *lo, T *hi)
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
FINUFFT_EXPORT void FINUFFT_CDECL arraywidcen(BIGINT n, T *a, T *w, T *c)
// Writes out w = half-width and c = center of an interval enclosing all a[n]'s
// Only chooses a nonzero center if this increases w by less than fraction
// ARRAYWIDCEN_GROWFRAC defined in defs.h.
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

#endif // UTILS_H
