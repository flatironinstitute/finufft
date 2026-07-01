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

// Upsampling-factor (sigma) "rails". There are deliberately TWO intervals, not one,
// because they serve two different jobs (a single interval cannot do both):
//
//  - [MIN_CHECK_SIGMA, MAX_CHECK_SIGMA] = [1.0, 2.0] is the ACCURACY rail used by
//    check_sigma/lowest_sigma: the range over which a sigma is allowed to be judged
//    able to reach a tol. Past 2.0 the kernel aliasing error barely improves (widening
//    ns is the better lever), so a tol that 2.0 cannot reach hits the rounding floor.
//    Lowering this ceiling re-trips check_sigma's floor on 3D (see PR #868 CI history).
//
//  - [MIN_AUTO_UPSAMPFAC, MAX_AUTO_UPSAMPFAC] = [1.15, 2.5] is the wider PERFORMANCE
//    search rail for the auto-heuristic (analytic_upsampfac / heuristics::minimize). A
//    narrower kernel (sigma up to 2.5) can still pay off for dense, spread-dominated
//    transforms even though it buys little extra accuracy.
//
// (The PR description's "[1.25, 2.00]" is just the range the heuristic happens to land
// in for typical inputs, not a third rail; it is not a constant.)
inline constexpr double MIN_CHECK_SIGMA = 1.0;
inline constexpr double MAX_CHECK_SIGMA = 2.0;

// Auto-heuristic (analytic_upsampfac / heuristics::minimize) search bounds.
inline constexpr double MIN_AUTO_UPSAMPFAC = 1.15;
inline constexpr double MAX_AUTO_UPSAMPFAC = 2.5;

// Single-precision catastrophic-cancellation guard (the ONE place these live).
// Below FLOAT_CC_UPSAMPFAC_LIMIT the dynamic range r_dyn blows up, so float kernels are
// capped to FLOAT_MAX_NS_CC to avoid losing accuracy. Referenced by clamp_kernel_ns
// (kernel.h) and the warning in makeplan.hpp; change here and both follow.
inline constexpr double FLOAT_CC_UPSAMPFAC_LIMIT = 1.4;
// max ns allowed (single prec, low sigma) without excessive catastrophic cancellation;
// hacky, const, found via tolsweeptest.m (type 3 was 7).
inline constexpr int FLOAT_MAX_NS_CC = 8;

} // namespace common
} // namespace finufft
