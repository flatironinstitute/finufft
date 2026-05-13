#pragma once

#include <type_traits>

namespace finufft {
namespace common {

// constants needed within common
// upper bound on w, i.e. nspread, even when padded.
// also for common
inline constexpr int MIN_NSPREAD = 2;
// Per-precision instantiation cap on ns.
// Single precision can never need ns>11 in practice: at the float epsilon
// (~1.19e-7) the kernel-width formulas in theoretical_kernel_ns and
// cufinufft's setup_spreader top out at 11 (kerformula=8, sigma>=1.4) and
// 12 (cufinufft, sigma=1.25); for sigma<1.4 makeplan additionally caps
// float to 8 to prevent catastrophic cancellation. We use 12 as a
// defensive margin. See issue #827.
template<typename T>
inline constexpr int MAX_NSPREAD = std::is_same_v<T, float> ? 12 : 16;
// max number of positive quadr nodes
inline constexpr int MAX_NQUAD = 100;
// Fraction growth cut-off in utils:arraywidcen, sets when translate in type-3
inline constexpr double ARRAYWIDCEN_GROWFRAC = 0.1;
inline constexpr double PI                   = 3.141592653589793238462643383279502884;
// 1 / (2 * PI)
inline constexpr double INV_2PI = 0.159154943091895335768883763372514362;

// polynomial degree bounds for all kernel approximation
inline constexpr int MIN_NC = 4;
inline constexpr int MAX_NC = 19;

// upsampling factor (sigma) bounds for the sigma_min estimator
inline constexpr double MINSIGMA = 1.0;
inline constexpr double MAXSIGMA = 2.0;

} // namespace common
} // namespace finufft
