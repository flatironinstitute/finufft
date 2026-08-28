#ifndef CUFINUFFT_TYPES_H
#define CUFINUFFT_TYPES_H

#include <cufft.h>
#include <cufinufft_opts.h>
#include <finufft_common/common.h>

#include <algorithm>
#include <cstdint>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <limits>

// FIXME: If cufft ever takes N > INT_MAX...
constexpr int32_t MAX_NF = std::numeric_limits<int32_t>::max();

using CUFINUFFT_BIGINT = int;

// Marco Barbone 8/5/2024, replaced the ugly trick with std::conditional
// to define cuda_complex
// by using std::conditional and std::is_same, we can define cuda_complex
// if T is float, cuda_complex<T> is cuFloatComplex
// if T is double, cuda_complex<T> is cuDoubleComplex
// where cuFloatComplex and cuDoubleComplex are defined in cuComplex.h
// TODO: migrate to cuda/std/complex and remove this
//       Issue: cufft seems not to support cuda::std::complex
//       A reinterpret_cast should be enough
template<typename T>
using cuda_complex = typename std::conditional<
    std::is_same<T, float>::value, cuFloatComplex,
    typename std::conditional<std::is_same<T, double>::value, cuDoubleComplex,
                              void>::type>::type;

template<typename T> static inline constexpr cufftType_t cufft_type();
template<> inline constexpr cufftType_t cufft_type<float>() { return CUFFT_C2C; }

template<> inline constexpr cufftType_t cufft_type<double>() { return CUFFT_Z2Z; }

static inline cufftResult cufft_ex(cufftHandle plan, cufftComplex *idata,
                                   cufftComplex *odata, int direction) {
  return cufftExecC2C(plan, idata, odata, direction);
}
static inline cufftResult cufft_ex(cufftHandle plan, cufftDoubleComplex *idata,
                                   cufftDoubleComplex *odata, int direction) {
  return cufftExecZ2Z(plan, idata, odata, direction);
}

// Device attributes the heuristics and launch paths need, queried once per plan with
// cudaDeviceGetAttribute. cudaGetDeviceProperties is far slower, so it stays in the
// debug print.
struct GpuCapabilities {
  int device_id{};
  int cc_major{}, cc_minor{};
  int max_smem_per_block_optin{}; // bytes
  int max_smem_per_sm{};          // bytes
  int max_threads_per_sm{};
  int multiprocessor_count{};
  int l2_cache_size{};    // bytes
  int memory_bus_width{}; // bits
  int memory_pools_supported{};

  static GpuCapabilities query(int device_id) {
    GpuCapabilities gpu{};
    gpu.device_id = device_id;
    const auto get = [device_id](int *dst, cudaDeviceAttr attr) {
      cudaDeviceGetAttribute(dst, attr, device_id);
    };
    get(&gpu.cc_major, cudaDevAttrComputeCapabilityMajor);
    get(&gpu.cc_minor, cudaDevAttrComputeCapabilityMinor);
    get(&gpu.max_smem_per_block_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin);
    get(&gpu.max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor);
    get(&gpu.max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor);
    get(&gpu.multiprocessor_count, cudaDevAttrMultiProcessorCount);
    get(&gpu.l2_cache_size, cudaDevAttrL2CacheSize);
    get(&gpu.memory_bus_width, cudaDevAttrGlobalMemoryBusWidth);
    get(&gpu.memory_pools_supported, cudaDevAttrMemoryPoolsSupported);
    return gpu;
  }

  int max_warps_per_sm() const { return max_threads_per_sm / 32; }

  // >= 512-bit bus: HBM datacenter parts plus 512-bit GDDR7 workstation Blackwell.
  // These keep large method-1 bins effective past L2 residency; 384-bit GDDR parts
  // (Ada, L40S) do not. See the method-1 binsize rule in heuristics.cu.
  bool wide_memory_bus() const { return memory_bus_width >= 512; }

  // Complex elements in the ~1/3 of L2 the batchsize heuristic budgets for the working
  // set (in + out + twiddle).
  template<typename T> std::int64_t l2_complex_budget() const {
    return l2_cache_size / 3 / std::int64_t(sizeof(cuda_complex<T>));
  }

  // Upper bound for "good enough" thread-block sizes:
  //   SM 9x / 8x : 16 warps = 256 threads
  //   SM 7x      :  8 warps = 128 threads
  //   SM 6x-     :  4 warps =  64 threads
  unsigned optimal_block_threads() const {
    if (cc_major >= 8) return 256;
    if (cc_major >= 7) return 128;
    return 64;
  }

  // Workload-aware block size for method-1 interpolation: optimal_block_threads() is
  // still the cap, but tiny transforms run better with fewer threads. Ramp up from one
  // warp in power-of-two steps until we hit the workload size or the device cap.
  unsigned interp_block_threads(int M) const {
    const auto limit =
        std::min(optimal_block_threads(), static_cast<unsigned>(std::max(M, 32)));
    if (limit >= 256) return 256;
    if (limit >= 128) return 128;
    if (limit >= 64) return 64;
    return 32;
  }

  // Defined in heuristics.cu: keeps <cstdio> out of this header.
  void print_classification(int debug_level) const;
};

#endif
