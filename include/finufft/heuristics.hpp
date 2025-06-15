#ifndef HEURISTICS_HPP
#define HEURISTICS_HPP

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace finufft::heuristics {
#ifndef FINUFFT_USE_DUCC0
template<typename T>
static double bestUpsamplingFactorSinglethread(
    const double density, const int dim, const int nufftType, const double epsilon) {

  constexpr bool isFloat = std::is_same_v<T, float>;

  if constexpr (isFloat) { // single precision
    if (nufftType == 1) {
      if (dim == 2 && density >= 32 && epsilon <= 1.0e-5) {
        return 2.0;
      }
      if (dim == 3 && density >= 2 && epsilon <= 1.0e-5) {
        return 2.0;
      }
    } // end nufftType == 1
    if (nufftType == 2) {
      if (dim == 1 && density >= 8 && epsilon <= 1.0e-5) {
        return 2.0;
      }
      if (dim == 3) {
        if (density >= 32 && epsilon <= 1.0e-3) {
          return 2.0;
        }
        if (density >= 4 && epsilon <= 1.0e-4) {
          return 2.0;
        }
        if (density >= 1 && epsilon <= 1.0e-5) {
          return 2.0;
        }
        return 2.0;
      } // end dim == 3
    } // end nufftType == 2
  } else { // end single precision, double precision follows
    if (nufftType == 1) {
      // interestingly for type 1, 1.25 is always faster in 1D
      if (dim == 2) {
        if (density >= 32 && epsilon <= 1.0e-6) {
          return 2.0;
        }
        if (density >= 16 && epsilon <= 1.0e-7) {
          return 2.0;
        }
        if (density >= 8 && epsilon <= 1.0e-8) {
          return 2.0;
        }
      } // end dim == 2
      if (dim == 3) {
        if (density >= 16 && epsilon <= 1.0e-3) {
          return 2.0;
        }
        if (density >= 2 && epsilon <= 1.0e-5) {
          return 2.0;
        }
        if (density >= 1 && epsilon <= 1.0e-7) {
          return 2.0;
        }
        if (epsilon <= 1.0e-8) { // for epsilon <= 1.0e-8, 2.0 is favored in 3D always
          return 2.0;
        }
      } // end dim == 3
    } // end nufftType == 1
    if (nufftType == 2) {
      if (dim == 1) {
        if (density >= 4 && epsilon <= 1.0e-9) {
          return 2.0;
        }
      } // end dim == 1
      if (dim == 2) {
        if (density >= 32 && epsilon <= 1.0e-7) {
          return 2.0;
        }
      } // end dim == 2
      if (dim == 3) {
        if (density >= 8 && epsilon <= 1.0e-3) {
          return 2.0;
        }
        if (density >= 4 && epsilon <= 1.0e-4) {
          return 2.0;
        }
        if (density >= 1 && epsilon <= 1.0e-5) {
          return 2.0;
        }
        if (density >= 0.5 && epsilon <= 1.0e-7) {
          return 2.0;
        }
        if (density >= 0.25 && epsilon <= 1.0e-8) {
          return 2.0;
        }
      } // end dim == 3
    } // end nufftType == 2
  }

  return 1.25;
}

template<typename T>
static double bestUpsamplingFactorMultithread(const double density, const int dim,
                                              const int nufftType, const double epsilon) {
  constexpr bool isFloat = std::is_same_v<T, float>;
  if constexpr (isFloat) {
    // in float multi-threaded, this is the only case where 2.0 is faster
    if (dim == 3 && nufftType == 1 && epsilon <= 1.0e-6 && density >= 32) {
      return 2.0;
    }
  } else {
    // In not 3D, 1.25 is always better
    if (dim != 3) return 1.25;

    if (nufftType == 1) {
      if (density >= 16 && epsilon <= 1.0e-5) {
        return 2.0;
      }
      if (density >= 8 && epsilon <= 1.0e-7) {
        return 2.0;
      }
      if (density >= 2 && epsilon <= 1.0e-8) {
        return 2.0;
      }
      if (density >= 1 && epsilon <= 1.0e-9) {
        return 2.0;
      }
    }
    if (nufftType == 2) {
      if (density >= 16 && epsilon <= 1.0e-5) {
        return 2.0;
      }
      if (density >= 8 && epsilon <= 1.0e-6) {
        return 2.0;
      }
      if (density >= 4 && epsilon <= 1.0e-7) {
        return 2.0;
      }
      if (density >= 2 && epsilon <= 1.0e-8) {
        return 2.0;
      }
      if (density >= 1 && epsilon <= 1.0e-9) {
        return 2.0;
      }
    }
  }

  return 1.25;
}

#else

template<typename T>
static double bestUpsamplingFactorSinglethread(
    const double density, const int dim, const int nufftType, const double epsilon) {

  constexpr bool isFloat = std::is_same_v<T, float>;

  if constexpr (isFloat) { // single precision
    if (nufftType == 1) {
      if (dim == 2 && density >= 4 && epsilon <= 1.0e-5) {
        return 2.0;
      }
      // if we got here and it is not 3D return 1.25
      if (dim != 3) {
        return 1.25;
      }
      if (density >= 16 && epsilon <= 1.0e-3) {
        return 2.0;
      }
      if (density >= 8 && epsilon <= 1.0e-4) {
        return 2.0;
      }
      if (density >= 1 && epsilon <= 1.0e-5) {
        return 2.0;
      }

    } // end nufftType == 1
    if (nufftType == 2) {
      if (dim == 1 && density >= 16 && epsilon <= 1.0e-5) {
        return 2.0;
      }
      // if we got here and it is not 3D return 1.25
      if (dim != 3) {
        return 1.25;
      }
      // 3D cases only here
      if (density >= 8 && epsilon <= 1.0e-3) {
        return 2.0;
      }
      if (density >= 2 && epsilon <= 1.0e-4) {
        return 2.0;
      }
      if (density >= 0.5 && epsilon <= 1.0e-5) {
        return 2.0;
      }
      if (density >= 0.25 && epsilon <= 1.0e-6) {
        return 2.0;
      }
    } // end nufftType == 2
  } else { // end single precision, double precision follows
    if (nufftType == 1) {
      // interestingly for type 1, 1.25 is always faster in 1D
      if (dim == 2) {
        if (density >= 8 && epsilon <= 1.0e-7) {
          return 2.0;
        }
      } // end dim == 2
      if (dim == 3) {
        if (density >= 16 && epsilon <= 1.0e-3) {
          return 2.0;
        }
        if (density >= 8 && epsilon <= 1.0e-4) {
          return 2.0;
        }
        if (density >= 2 && epsilon <= 1.0e-5) {
          return 2.0;
        }
        if (density >= 1 && epsilon <= 1.0e-6) {
          return 2.0;
        }
        if (density >= 0.5 && epsilon <= 1.0e-7) {
          return 2.0;
        }
        if (density >= 0.25 && epsilon <= 1.0e-8) {
          return 2.0;
        }
        if (density >= 0.125 && epsilon <= 1.0e-9) {
          return 2.0;
        }
      } // end dim == 3
    } // end nufftType == 1
    if (nufftType == 2) {
      if (dim == 1) {
        return 1.25;
      } // end dim == 1
      if (dim == 2) {
        if (density >= 16 && epsilon <= 1.0e-7) {
          return 2.0;
        }
      } // end dim == 2
      if (dim == 3) {
        if (density >= 8 && epsilon <= 1.0e-3) {
          return 2.0;
        }
        if (density >= 2 && epsilon <= 1.0e-4) {
          return 2.0;
        }
        if (density >= 1 && epsilon <= 1.0e-5) {
          return 2.0;
        }
        if (density >= 0.5 && epsilon <= 1.0e-6) {
          return 2.0;
        }
        if (density >= 0.25 && epsilon <= 1.0e-8) {
          return 2.0;
        }
        if (density >= 0.125 && epsilon <= 1.0e-9) {
          return 2.0;
        }
      } // end dim == 3
    } // end nufftType == 2
  }
  return 1.25;
}

template<typename T>
static double bestUpsamplingFactorMultithread(const double density, const int dim,
                                              const int nufftType, const double epsilon) {
  constexpr bool isFloat = std::is_same_v<T, float>;
  if constexpr (isFloat) {
    if (dim == 3 && nufftType == 1 && epsilon <= 1.0e-5 && density >= 16) {
      return 2.0;
    }
    if (dim == 3 && nufftType == 2 && epsilon <= 1.0e-5 && density >= 8) {
      return 2.0;
    }

  } else {
    // only case where 2.0 is faster in 2D
    if (dim == 2 && nufftType == 1 && density >= 32 && epsilon <= 1.0e-8) {
      return 2.0;
    }
    // the following are only 3D cases if it reaches here and is not 3D return 1.25
    if (dim != 3) {
      return 1.25;
    }
    if (nufftType == 1) { // same as fftw here
      if (density >= 16 && epsilon <= 1.0e-5) {
        return 2.0;
      }
      if (density >= 8 && epsilon <= 1.0e-7) {
        return 2.0;
      }
      if (density >= 2 && epsilon <= 1.0e-8) {
        return 2.0;
      }
      if (density >= 1 && epsilon <= 1.0e-9) {
        return 2.0;
      }
    }
    if (nufftType == 2) { // some different cases here
      if (density >= 8 && epsilon <= 1.0e-5) {
        return 2.0;
      }
      if (density >= 4 && epsilon <= 1.0e-7) {
        return 2.0;
      }
      if (density >= 1 && epsilon <= 1.0e-8) {
        return 2.0;
      }
    }
  }

  return 1.25;
}

#endif
template<typename T>
double bestUpsamplingFactor(const int nthreads, const double density, const int dim,
                            const int nufftType, const double epsilon) {
  // 1) For epsilons <= 1e-9, 1.25 is not supported.
  //    We also prevent 1.25 being used when within 2 digits of eps_mach
  if (epsilon <= 1.0e-9 || epsilon <= std::numeric_limits<T>::epsilon() * 100) {
    return 2.0;
  }

  // 2) Special-case for nufftType == 3
  //    TODO: maybe use the bandwidth here?
  if (nufftType == 3) {
    return 1.25;
  }
  // 3) For 1 thread, use single-threaded heuristic
  if (nthreads == 1) {
    return bestUpsamplingFactorSinglethread<T>(density, dim, nufftType, epsilon);
  }
  // 4) For 2 threads, use multi-threaded heuristic
  if (nthreads > 1) {
    return bestUpsamplingFactorMultithread<T>(density, dim, nufftType, epsilon);
  }

  // 4) Otherwise, return 1.25
  return 1.25;
}

} // end namespace finufft::heuristics

#endif // HEURISTICS_HPP
