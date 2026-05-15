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

#include <string>

#include <finufft_common/safe_call.h>
#include <finufft_errors.h>

#include <cuda_runtime.h>
#include <cufft.h>

namespace cufinufft {

inline const char *cufftGetErrorString(cufftResult error) {
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

  case CUFFT_INVALID_DEVICE:
    return "CUFFT_INVALID_DEVICE";

  case CUFFT_NO_WORKSPACE:
    return "CUFFT_NO_WORKSPACE";

  case CUFFT_NOT_IMPLEMENTED:
    return "CUFFT_NOT_IMPLEMENTED";

  case CUFFT_NOT_SUPPORTED:
    return "CUFFT_NOT_SUPPORTED";

// Deprecated in cuFFT 12.9, removed in cuFFT 13. Only reference them when
// compiling with nvcc < 13 (where the enums exist). `__CUDACC_VER_MAJOR__`
// is undefined when this header is included from a plain C++ TU, in which
// case the legacy cases are skipped too.
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ < 13
  case CUFFT_LICENSE_ERROR:
    return "CUFFT_LICENSE_ERROR";

  case CUFFT_PARSE_ERROR:
    return "CUFFT_PARSE_ERROR";

  case CUFFT_INCOMPLETE_PARAMETER_LIST:
    return "CUFFT_INCOMPLETE_PARAMETER_LIST";
#endif

  default:
    break; // Unknown / vendor-specific codes fall through.
  }
  return "<unknown>";
}

// Typed exception carrying a cudaError_t plus a caller-supplied operation tag
// and optional source location. Mirrors how Thrust / CCCL / CUTLASS / PyTorch
// / cuda-api-wrappers surface CUDA failures: a subclass with the raw status
// code and a context string.
class cuda_exception final : public finufft::exception {
public:
  cuda_exception(cudaError_t err, const char *op, const char *file = nullptr,
                 int line = 0)
      : finufft::exception(FINUFFT_ERR_CUDA_FAILURE, format(err, op, file, line)),
        cuda_code_(err) {}

  cudaError_t cuda_code() const noexcept { return cuda_code_; }

private:
  cudaError_t cuda_code_;

  static std::string format(cudaError_t e, const char *op, const char *file, int line) {
    std::string s = op ? op : "<unknown>";
    if (file) {
      s += " @ ";
      s += file;
      s += ":";
      s += std::to_string(line);
    }
    s += ": ";
    s += cudaGetErrorName(e);
    s += " (";
    s += cudaGetErrorString(e);
    s += ")";
    return s;
  }
};

class cufft_exception final : public finufft::exception {
public:
  cufft_exception(cufftResult err, const char *op, const char *file = nullptr,
                  int line = 0)
      : finufft::exception(FINUFFT_ERR_CUDA_FAILURE, format(err, op, file, line)),
        cufft_code_(err) {}

  cufftResult cufft_code() const noexcept { return cufft_code_; }

private:
  cufftResult cufft_code_;

  static std::string format(cufftResult e, const char *op, const char *file, int line) {
    std::string s = op ? op : "<unknown>";
    if (file) {
      s += " @ ";
      s += file;
      s += ":";
      s += std::to_string(line);
    }
    s += ": ";
    s += cufftGetErrorString(e);
    return s;
  }
};

namespace detail {

// Sample (and clear) the sticky CUDA error flag. Throws if a kernel launch or
// prior async call left an error on the stream. The macro form below fills
// `op` from `__func__` and `file`/`line` from the call site.
inline void throw_if_cuda_error(const char *op, const char *file = nullptr,
                                int line = 0) {
  if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
    throw cuda_exception(e, op, file, line);
}

// Check a synchronous CUDA runtime return code. Used by checkCudaErrors().
inline void check_cuda_status(cudaError_t s, const char *op, const char *file = nullptr,
                              int line = 0) {
  if (s != cudaSuccess) throw cuda_exception(s, op, file, line);
}

} // namespace detail
} // namespace cufinufft

// Wrap a CUDA runtime call; `#call` stringifies the expression for the tag and
// `__FILE__`/`__LINE__` pinpoint the call site for debugging.
#define checkCudaErrors(call) \
  ::cufinufft::detail::check_cuda_status((call), #call, __FILE__, __LINE__)

// Sample the sticky CUDA error flag and tag it with the enclosing function
// name and source location. `__func__` (C++11) and `__FILE__`/`__LINE__` are
// the standard portable way to capture caller context inside a macro.
#define THROW_IF_CUDA_ERROR() \
  ::cufinufft::detail::throw_if_cuda_error(__func__, __FILE__, __LINE__)

#endif // COMMON_HELPER_CUDA_H_
