#ifndef __MEMTRANSFER_H__
#define __MEMTRANSFER_H__

#include "cufinufft/types.h"

namespace cufinufft {
namespace memtransfer {

template<typename T> void allocgpumem1d_plan(cufinufft_plan_t<T> *d_plan);
template<typename T> void allocgpumem1d_nupts(cufinufft_plan_t<T> *d_plan);
template<typename T> void freegpumemory(cufinufft_plan_t<T> *d_plan);
template<typename T> void allocgpumem2d_plan(cufinufft_plan_t<T> *d_plan);
template<typename T> void allocgpumem2d_nupts(cufinufft_plan_t<T> *d_plan);
template<typename T> void allocgpumem3d_plan(cufinufft_plan_t<T> *d_plan);
template<typename T> void allocgpumem3d_nupts(cufinufft_plan_t<T> *d_plan);

} // namespace memtransfer
} // namespace cufinufft
#endif
