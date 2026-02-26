#ifndef __CUDECONVOLVE_H__
#define __CUDECONVOLVE_H__

#include <cufinufft/types.h>

namespace cufinufft {
namespace deconvolve {

template<typename T>
void cudeconvolve(cufinufft_plan_t<T> *d_mem, int blksize);

#if 0
template<typename T, int modeord>
void cudeconvolve1d(cufinufft_plan_t<T> *d_mem, int blksize);
template<typename T, int modeord>
void cudeconvolve2d(cufinufft_plan_t<T> *d_mem, int blksize);
template<typename T, int modeord>
void cudeconvolve3d(cufinufft_plan_t<T> *d_mem, int blksize);
#endif
} // namespace deconvolve
} // namespace cufinufft
#endif
