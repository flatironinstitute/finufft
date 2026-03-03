#ifndef CUFINUFFT_DEFS_H
#define CUFINUFFT_DEFS_H

#include <finufft_common/common.h>
#include <limits>

// FIXME: If cufft ever takes N > INT_MAX...
constexpr int32_t MAX_NF = std::numeric_limits<int32_t>::max();

#endif
