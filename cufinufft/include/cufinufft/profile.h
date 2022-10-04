#ifndef PROFILE_H
#define PROFILE_H

#include <stdint.h>

namespace cufinufft {
namespace profile {
class CudaTracer {
  public:
    CudaTracer(const char *name, int cid = 0);
    ~CudaTracer();
};
} // namespace profile
} // namespace cufinufft

#define PROFILE_CUDA(fname) cufinufft::profile::CudaTracer uniq_name_using_macros__(fname);
#define PROFILE_CUDA_GROUP(fname, groupid) cufinufft::profile::CudaTracer uniq_name_using_macros__(fname, groupid);

#endif
