#pragma once

#include <type_traits>

namespace finufft {
namespace common {
// ------------------- Constants needed within common -----------------------

// Smallest spread width ns
inline constexpr int MIN_NSPREAD = 2;

// Largest spread width ns. (Note: above sigma 1.7 the float error is flat for
// ns 7-16, so with float, ns > 12 gains nothing.)
template<typename T>
inline constexpr int MAX_NSPREAD = std::is_same_v<T, float> ? 12 : 16;

// max number of positive quadrature nodes for kernel FT onedim_fseries_kernel
inline constexpr int MAX_NQUAD = 100;

// Fraction growth cut-off in utils:arraywidcen, sets when to translate type-3
inline constexpr double ARRAYWIDCEN_GROWFRAC = 0.1;

// Math constants...
inline constexpr double PI                       = 3.141592653589793238462643383279502884;
inline constexpr double INV_2PI = 0.159154943091895335768883763372514362;

// polynomial degree bounds for all kernel Horner approximation
inline constexpr int MIN_NC = 4;
inline constexpr int MAX_NC = 19;

// Auto-choice of upsampfac in best_sigma_for_tol() searches in this range:
// (Notes: below 1.25 it lands on the aliasing-law boundary with no accuracy
// margin; above 2.0 a narrower kernel can still pay off for dense,
// spread-dominated transforms)
inline constexpr double MIN_AUTO_UPSAMPFAC       = 1.25;
inline constexpr double MAX_AUTO_UPSAMPFAC       = 2.5;

// Makeplan warns on a user-chosen upsampfac outside of this (wider) range:
inline constexpr double MIN_USEFUL_UPSAMPFAC     = 1.15;
inline constexpr double MAX_USEFUL_UPSAMPFAC     = 3.0;

// How much sigma gets bumped up to be safely on the correct side of the
// boundary induced by the ceil in theoretical_kernel_ns. Must also be
// small enough to neither materially reduce efficiency nor cause ns to
// decrease to the next integer...
inline constexpr double UPSAMPFAC_EDGE_PAD       = 1e-3;

// Width (ns) offset in the kernel aliasing law, used by
// theoretical_kernel_ns() and its inverse min_sigma_for_aliasing_error()
inline constexpr double NSOFF                    = 1.0;

// Fitted prefactor for condition-number associated rounding_error()
// (in src/common/kernel.cpp)
inline constexpr double ROUND_FAC                = 0.35;

} // namespace common
} // namespace finufft
