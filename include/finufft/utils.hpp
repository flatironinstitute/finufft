// Header for utils.cpp, a little library of low-level array stuff.
// These are just the functions which depend on single/double precision (FLT)

#pragma once

#include <cmath>
#include <cstddef>
#include <finufft_common/common.h>

namespace finufft::utils {

// arrayrange / arraywidcen now live in finufft_common/utils.h (BIGINT is a
// typedef for int64_t, which the shared decl uses).

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

// Bytes of the L1 data cache and of the L2 cache on one core, queried once and cached.
// Fall back to 32 KiB and 2 MiB on a platform that does not report them.
FINUFFT_EXPORT_TEST FINUFFT_NEVER_INLINE std::size_t getL1CacheSize() noexcept;
FINUFFT_EXPORT_TEST FINUFFT_NEVER_INLINE std::size_t getL2CacheSize() noexcept;

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
