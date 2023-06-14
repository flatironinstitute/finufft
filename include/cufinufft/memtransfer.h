#ifndef __MEMTRANSFER_H__
#define __MEMTRANSFER_H__

#include "cufinufft/types.h"

namespace cufinufft {
namespace memtransfer {

template <typename T>
int allocgpumem1d_plan(cufinufft_plan_template<T> d_plan);
template <typename T>
int allocgpumem1d_nupts(cufinufft_plan_template<T> d_plan);
template <typename T>
void freegpumemory1d(cufinufft_plan_template<T> d_plan);
template <typename T>
int allocgpumem2d_plan(cufinufft_plan_template<T> d_plan);
template <typename T>
int allocgpumem2d_nupts(cufinufft_plan_template<T> d_plan);
template <typename T>
void freegpumemory2d(cufinufft_plan_template<T> d_plan);
template <typename T>
int allocgpumem3d_plan(cufinufft_plan_template<T> d_plan);
template <typename T>
int allocgpumem3d_nupts(cufinufft_plan_template<T> d_plan);
template <typename T>
void freegpumemory3d(cufinufft_plan_template<T> d_plan);

} // namespace mem
} // namespace cufinufft
#endif
