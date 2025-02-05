#ifndef HEURISTICS_HPP
#define HEURISTICS_HPP

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace finufft::heuristics {

// bestUpsamplingFactor()
// Template parameter T (float or double) is used to branch on precision.
// Function to determine the best upsampling factor.
// Returns either 1.25 or 2.0
template<typename T>
double bestUpsamplingFactor(double density, int dim, int nufftType, double epsilon) {
  // 1) For epsilons <= 1e-9, 2.0 is favored
  if (epsilon <= 1.0e-9) {
    return 2.0;
  }

  // 2) Special-case for nufftType == 3
  if (nufftType == 3) {
    return 1.25;
  }

  // 3) High density check
  if (density > 100.0) {
    return 2.0;
  }

  // 4) If density > 10, check dimension-based thresholds
  constexpr bool isFloat = std::is_same_v<T, float>;
  if (density > 10.0) {
    if constexpr (isFloat) {
      if (dim == 1 && epsilon <= 1.0e-3) return 2.0;
      if (dim == 2 && epsilon <= 1.0e-5) return 2.0;
      if (dim == 3 && epsilon <= 1.0e-6) return 2.0;
    } else {
      if (dim == 1 && epsilon <= 1.0e-7) return 2.0;
      if (dim == 2 && epsilon <= 1.0e-8) return 2.0;
      if (dim == 3 && epsilon <= 1.0e-8) return 2.0;
    }
  }

  // 5) Otherwise, return 1.25
  return 1.25;
}
} // end namespace finufft::heuristics

#endif // HEURISTICS_HPP
