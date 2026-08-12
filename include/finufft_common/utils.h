#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include "constants.h"
#include "defines.h"

// Runtime->compile-time dispatch (formerly make_range/DispatchParam/dispatch,
// defined here) now comes from POET, included only in the TUs that dispatch:
//   make_range    -> poet::inclusive_range
//   DispatchParam -> poet::dispatch_param
//   dispatch      -> poet::dispatch

namespace finufft {
namespace common {

// Returns p*required factor, where p is the smallest composite
// of 2, 3, 5 such that p*required_factor >= n.
long next235(long n, long required_factor = 1);

FINUFFT_EXPORT_TEST void gaussquad(int n, double *xgl, double *wgl);
std::tuple<double, double> leg_eval(int n, double x);

// Series implementation of the modified Bessel function of the first kind I_nu(x)
double cyl_bessel_i(double nu, double x) noexcept;
// Explicit custom series implementation exposed for testing
double cyl_bessel_i_custom(double nu, double x) noexcept;

} // namespace common
} // namespace finufft

namespace finufft::utils {

// Round v down / up to a multiple of align. Every align used here is a SIMD
// width or a kernel-buffer stride, i.e. a power of two; that is the property
// the mask form relies on, so it is checked rather than assumed. The runtime
// overloads exist so that a compile-time formula and its runtime mirror cannot
// drift apart (see kernel_buffer_stride_runtime in simd.hpp).
template<typename T> constexpr T round_down(T v, std::size_t align) noexcept {
  assert(align != 0 && (align & (align - 1)) == 0); // power of two
  return v & ~T(align - 1);
}
template<typename T> constexpr T round_up(T v, std::size_t align) noexcept {
  return round_down(T(v + T(align - 1)), align);
}
template<std::size_t Align, typename T> constexpr T round_down(T v) noexcept {
  static_assert(Align != 0 && (Align & (Align - 1)) == 0, "Align must be a power of two");
  static_assert(Align - 1 <= std::size_t((std::numeric_limits<T>::max)()),
                "Align does not fit in T; the mask would wrap");
  return round_down(v, Align);
}
template<std::size_t Align, typename T> constexpr T round_up(T v) noexcept {
  return round_down<Align>(T(v + T(Align - 1)));
}

// Host versions of arrayrange / arraywidcen. The CUDA path has a separate
// device-pointer overload (in include/cufinufft/utils.hpp) that uses thrust.
template<typename T>
FINUFFT_ALWAYS_INLINE void arrayrange(int64_t n, const T *a, T *lo, T *hi)
// With a a length-n array, writes out min(a) to lo and max(a) to hi,
// so that all a values lie in [lo,hi].
// If n==0, lo and hi are not finite.
{
  *lo = INFINITY;
  *hi = -INFINITY;
  for (int64_t m = 0; m < n; ++m) {
    if (a[m] < *lo) *lo = a[m];
    if (a[m] > *hi) *hi = a[m];
  }
}
template<typename T>
FINUFFT_ALWAYS_INLINE void arraywidcen(int64_t n, const T *a, T *w, T *c)
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
  if (std::abs(*c) < finufft::common::ARRAYWIDCEN_GROWFRAC * (*w)) {
    *w += std::abs(*c);
    *c = 0.0;
  }
}

} // namespace finufft::utils
