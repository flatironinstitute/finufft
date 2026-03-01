#ifndef __MEMTRANSFER_H__
#define __MEMTRANSFER_H__

#include "cufinufft/types.h"

namespace cufinufft {
namespace memtransfer {

template<typename T> void allocgpumem_plan(cufinufft_plan_t<T> &d_plan);
template<typename T> void allocgpumem_nupts(cufinufft_plan_t<T> &d_plan);

} // namespace memtransfer
} // namespace cufinufft
#endif
