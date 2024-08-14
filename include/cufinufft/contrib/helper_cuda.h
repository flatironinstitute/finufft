/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef COMMON_HELPER_CUDA_H_
#define COMMON_HELPER_CUDA_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <finufft_errors.h>

#include <cufft.h>

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template<typename T>
static inline cudaError_t cudaMallocWrapper(T **devPtr, size_t size, cudaStream_t stream,
                                            int pool_supported) {
  return pool_supported ? cudaMallocAsync(devPtr, size, stream)
                        : cudaMalloc(devPtr, size);
}

template<typename T>
static inline cudaError_t cudaFreeWrapper(T *devPtr, cudaStream_t stream,
                                          int pool_supported) {
  return pool_supported ? cudaFreeAsync(devPtr, stream) : cudaFree(devPtr);
}

#define RETURN_IF_CUDA_ERROR                                                         \
  {                                                                                  \
    cudaError_t err = cudaGetLastError();                                            \
    if (err != cudaSuccess) {                                                        \
      printf("[%s] Error: %s in %s at line %d\n", __func__, cudaGetErrorString(err), \
             __FILE__, __LINE__);                                                    \
      return FINUFFT_ERR_CUDA_FAILURE;                                               \
    }                                                                                \
  }

#define CUDA_FREE_AND_NULL(val, stream, pool_supported)                              \
  {                                                                                  \
    if (val != nullptr) {                                                            \
      check(cudaFreeWrapper(val, stream, pool_supported), #val, __FILE__, __LINE__); \
      val = nullptr;                                                                 \
    }                                                                                \
  }

static const char *cufftGetErrorString(cufftResult error) {
  switch (error) {
  case CUFFT_SUCCESS:
    return "CUFFT_SUCCESS";

  case CUFFT_INVALID_PLAN:
    return "CUFFT_INVALID_PLAN";

  case CUFFT_ALLOC_FAILED:
    return "CUFFT_ALLOC_FAILED";

  case CUFFT_INVALID_TYPE:
    return "CUFFT_INVALID_TYPE";

  case CUFFT_INVALID_VALUE:
    return "CUFFT_INVALID_VALUE";

  case CUFFT_INTERNAL_ERROR:
    return "CUFFT_INTERNAL_ERROR";

  case CUFFT_EXEC_FAILED:
    return "CUFFT_EXEC_FAILED";

  case CUFFT_SETUP_FAILED:
    return "CUFFT_SETUP_FAILED";

  case CUFFT_INVALID_SIZE:
    return "CUFFT_INVALID_SIZE";

  case CUFFT_UNALIGNED_DATA:
    return "CUFFT_UNALIGNED_DATA";

  case CUFFT_INCOMPLETE_PARAMETER_LIST:
    return "CUFFT_INCOMPLETE_PARAMETER_LIST";

  case CUFFT_INVALID_DEVICE:
    return "CUFFT_INVALID_DEVICE";

  case CUFFT_PARSE_ERROR:
    return "CUFFT_PARSE_ERROR";

  case CUFFT_NO_WORKSPACE:
    return "CUFFT_NO_WORKSPACE";

  case CUFFT_NOT_IMPLEMENTED:
    return "CUFFT_NOT_IMPLEMENTED";

  case CUFFT_LICENSE_ERROR:
    return "CUFFT_LICENSE_ERROR";

  case CUFFT_NOT_SUPPORTED:
    return "CUFFT_NOT_SUPPORTED";
  }

  return "<unknown>";
}

template<typename T>
int check(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    return FINUFFT_ERR_CUDA_FAILURE;
  }

  return 0;
}

#endif // COMMON_HELPER_CUDA_H_
