#ifndef __CUDECONVOLVE_H__
#define __CUDECONVOLVE_H__

#include <cufinufft/types.h>

namespace cufinufft {
namespace deconvolve {

template<typename T> void cudeconvolve(const cufinufft_plan_t<T> &d_plan, int blksize);

} // namespace deconvolve
} // namespace cufinufft
#endif
