#pragma once

#include <type_traits>

namespace finufft {
namespace common {

// constants needed within common
// Smallest spread width ns.
inline constexpr int MIN_NSPREAD = 2;
// Per-precision instantiation cap on ns. Above sigma 1.7 the float error is flat for
// ns 7-16, so float ns > 12 gains nothing.
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

// Upsampfac (sigma) bounds.
// The auto pick searches [MIN_AUTO_UPSAMPFAC, MAX_AUTO_UPSAMPFAC] for all types. Below
// 1.25 it lands on the aliasing-law boundary with no accuracy margin; above 2.0 a
// narrower kernel can still pay off for dense, spread-dominated transforms.
inline constexpr double MIN_AUTO_UPSAMPFAC       = 1.25;
inline constexpr double MAX_AUTO_UPSAMPFAC       = 2.5;
// makeplan warns on a user upsampfac outside this range, which rarely pays off.
inline constexpr double MIN_USEFUL_UPSAMPFAC     = 1.15;
inline constexpr double MAX_USEFUL_UPSAMPFAC     = 3.0;

// eps_round = ROUND_FAC eps_mach N sqrt(dim) (rounding_floor, kernel.h).
inline constexpr double ROUND_FAC                = 0.35;

} // namespace common
} // namespace finufft
