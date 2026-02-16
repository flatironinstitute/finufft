#ifndef __CUDECONVOLVE_H__
#define __CUDECONVOLVE_H__

#include <cufinufft/types.h>

namespace cufinufft {
namespace deconvolve {

template<typename T, int modeord>
int cudeconvolve1d(cufinufft_plan_t<T> *d_mem, int blksize);
template<typename T, int modeord>
int cudeconvolve2d(cufinufft_plan_t<T> *d_mem, int blksize);
template<typename T, int modeord>
int cudeconvolve3d(cufinufft_plan_t<T> *d_mem, int blksize);

} // namespace deconvolve
} // namespace cufinufft
#endif
