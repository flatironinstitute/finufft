// Header for utils.cpp, a little library of low-level array stuff.
// These are just the functions which depend on single/double precision (FLT)

#pragma once

#include "finufft_core.h"
#include <cmath>
#include <cstddef>
#include <finufft_common/common.h>
#if __has_include(<xsimd/xsimd.hpp>)
#include <array>
#include <finufft/xsimd.hpp>
#include <type_traits>
#if defined(XSIMD_VERSION_MAJOR) && (XSIMD_VERSION_MAJOR >= 14)

namespace finufft::utils {

template<class T, uint8_t N = 1> constexpr uint8_t min_simd_width() {
  // finds the smallest simd width that can handle N elements
  // simd size is batch size the SIMD width in xsimd terminology
  if constexpr (std::is_void_v<xsimd::make_sized_batch_t<T, N>>) {
    return min_simd_width<T, N * 2>();
  } else {
    return N;
  }
};

template<class T, uint8_t N> constexpr std::size_t find_optimal_simd_width() {
  // finds the smallest simd width that minimizes the number of iterations
  // NOTE: might be suboptimal for some cases 2^N+1 for example
  // in the future we might want to implement a more sophisticated algorithm

  uint8_t optimal_simd_width = min_simd_width<T>();
  uint8_t min_iterations     = (N + optimal_simd_width - 1) / optimal_simd_width;
  for (uint8_t simd_width = optimal_simd_width;
       simd_width <= xsimd::batch<T, xsimd::best_arch>::size; simd_width *= 2) {
    uint8_t iterations = (N + simd_width - 1) / simd_width;
    if (iterations < min_iterations) {
      min_iterations     = iterations;
      optimal_simd_width = simd_width;
    }
  }
  return static_cast<std::size_t>(optimal_simd_width);
}

template<class T, uint8_t N> constexpr std::size_t GetPaddedSIMDWidth() {
  // helper function to get the SIMD width with padding for the given number of elements
  // that minimizes the number of iterations

  return xsimd::make_sized_batch<T, find_optimal_simd_width<T, N>()>::type::size;
}
template<class T, uint8_t ns>
constexpr std::size_t get_simd_width_helper(uint8_t runtime_ns) {
  if constexpr (ns < finufft::common::MIN_NSPREAD) {
    return static_cast<std::size_t>(0);
  } else {
    if (runtime_ns == ns) {
      return GetPaddedSIMDWidth<T, ns>();
    } else {
      return get_simd_width_helper<T, ns - 1>(runtime_ns);
    }
  }
}
template<class T> constexpr std::size_t GetPaddedSIMDWidth(int runtime_ns) {
  return get_simd_width_helper<T, 2 * ::finufft::common::MAX_NSPREAD>(runtime_ns);
}

} // namespace finufft::utils
#endif // XSIMD_VERSION_MAJOR >=14
#endif // __has_include(xsimd)

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
FINUFFT_EXPORT_TEST BIGINT next235even(BIGINT n);
// jfm's timer class
class FINUFFT_EXPORT_TEST CNTime {
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
FINUFFT_EXPORT_TEST int rand_r(unsigned int *seedp);
} // namespace utils
} // namespace finufft
#endif
