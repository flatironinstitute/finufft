#pragma once

namespace finufft {
namespace common {

// constants needed within common
// upper bound on w, ie nspread, even when padded (see evaluate_kernel_vector);
// also for common
inline constexpr int MIN_NSPREAD = 2;
inline constexpr int MAX_NSPREAD = 16;
// max number of positive quadr nodes
inline constexpr int MAX_NQUAD = 100;
// Fraction growth cut-off in utils:arraywidcen, sets when translate in type-3
inline constexpr double ARRAYWIDCEN_GROWFRAC = 0.1;
// How many waays there are to evaluate the kernel it should match the avbailable options
// in finufft_opts
inline constexpr int KEREVAL_METHODS = 2;
inline constexpr double PI           = 3.141592653589793238462643383279502884;
// 1 / (2 * PI)
inline constexpr double INV_2PI = 0.159154943091895335768883763372514362;
} // namespace common
} // namespace finufft
