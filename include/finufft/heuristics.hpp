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
// Returns either 1.25 or 2.0 based on a heuristic that depends on:
//   - volume: the grid volume (nx * ny * nz),
//   - numPts: total number of nonuniform points,
//   - dim: grid dimension (1 for 1D, 2 for 2D, 3 for 3D),
//   - nufftType: typically 1 or 2 (with a special case for type 3),
//   - epsilon: the requested accuracy (always passed as double).
//
// The density is computed as density = numPts / volume.
// For 2D and 3D grids, a grid is considered "high density" if density > 100.0.
//
// Template parameter T (float or double) is used to branch on precision.
template<typename T>
double bestUpsamplingFactor(int volume, int numPts, int dim, int nufftType,
                            double epsilon) {
  // For epsilons <= 1e-9, return 2.0.
  if (epsilon <= 1e-9) {
    return 2.0;
  }

  // Special-case: if nufftType == 3, always return 1.25.
  if (nufftType == 3 && epsilon >= 1e-9) {
    return 1.25;
  }

  // Factor out the precision check.
  constexpr bool isFloat = std::is_same_v<T, float>;

  const double density = double(numPts) / volume;

  // For very low density, always choose 1.25.
  if (density < 1.0) {
    return 1.25;
  }

  // For 2D and 3D, decide if the grid is "high density" based solely on density.
  bool highDensity = false;
  if (dim >= 2) {
    highDensity = (density > 100.0);
  }

  if constexpr (isFloat) {
    // --- All float32 cases ---
    if (nufftType == 1) {
      if (dim == 1) {
        if (std::fabs(epsilon - 1e-03) < 1e-6) {
          return 2.0;
        } else {
          return 1.25;
        }
      } else if (dim == 2) {
        if (epsilon <= 1e-05) {
          return 2.0;
        } else {
          return 1.25;
        }
      } else if (dim == 3) {
        if (highDensity) {
          if (epsilon < 1e-03) {
            return 2.0;
          } else {
            return 1.25;
          }
        } else {
          if (epsilon <= 1e-04) {
            return 2.0;
          } else {
            return 1.25;
          }
        }
      }
    } else if (nufftType == 2) {
      if (dim == 1) {
        if (epsilon <= 1e-05) {
          return 2.0;
        } else {
          return 1.25;
        }
      } else if (dim == 2) {
        if (highDensity) {
          if (epsilon < 1e-05) {
            return 2.0;
          } else {
            return 1.25;
          }
        } else {
          return 1.25;
        }
      } else if (dim == 3) {
        if (highDensity) {
          // Fix: use <= for epsilon in high-density 3D Type 2.
          if (epsilon <= 1e-03) {
            return 2.0;
          } else {
            return 1.25;
          }
        } else {
          if (epsilon <= 1e-04) {
            return 2.0;
          } else {
            return 1.25;
          }
        }
      }
    }
  } else {
    // --- All double cases ---
    if (nufftType == 1) {
      if (dim == 1) {
        return 1.25;
      } else if (dim == 2) {
        if (highDensity) {
          if (epsilon <= 1e-05) {
            return 2.0;
          } else {
            return 1.25;
          }
        } else {
          if (epsilon <= 1e-07) {
            return 2.0;
          } else {
            return 1.25;
          }
        }
      } else if (dim == 3) {
        if (highDensity) {
          if (epsilon <= 1e-02) {
            return 2.0;
          } else {
            return 1.25;
          }
        } else {
          if (epsilon <= 1e-04) {
            return 2.0;
          } else {
            return 1.25;
          }
        }
      }
    } else if (nufftType == 2) {
      if (dim == 1) {
        if (epsilon <= 1e-09) {
          return 2.0;
        } else {
          return 1.25;
        }
      } else if (dim == 2) {
        // For double 2D normal, if not high density then use <= 1e-08.
        if (!highDensity) {
          if (epsilon <= 1e-08) {
            return 2.0;
          } else {
            return 1.25;
          }
        } else {
          if (epsilon <= 1e-05) {
            return 2.0;
          } else {
            return 1.25;
          }
        }
      } else if (dim == 3) {
        if (highDensity) {
          if (epsilon <= 1e-02) {
            return 2.0;
          } else {
            return 1.25;
          }
        } else {
          if (epsilon <= 1e-03) {
            return 2.0;
          } else {
            return 1.25;
          }
        }
      }
    }
  }
  return 1.25; // Fallback.
}

} // end namespace finufft::heuristics

#endif // HEURISTICS_HPP
