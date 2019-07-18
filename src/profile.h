#ifndef PROFILE_H
#define PROFILE_H

#include <stdint.h>

class CudaTracer {
    public:
        CudaTracer(const char* name, int cid = 0);
        ~CudaTracer();
};



#define PROFILE_CUDA(fname) CudaTracer uniq_name_using_macros__(fname);
#define PROFILE_CUDA_GROUP(fname, groupid) CudaTracer uniq_name_using_macros__(fname, groupid);

#endif
