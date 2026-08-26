#ifndef FINUFFT_UTILS_H
#define FINUFFT_UTILS_H

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <complex>

#include <cufinufft/types.hpp>

#include <cuda_runtime.h>
#include <thrust/extrema.h>
#include <tuple>
#include <type_traits>
#include <utility> // for std::forward

#include <finufft_common/common.h>

#include <poet/poet.hpp> // poet::dispatch / inclusive_range / dispatch_param

#include <cmath> // std math only; finufft uses finufft::common::PI, not M_PI

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600 || defined(__clang__)
#else
__inline__ __device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old             = *address_as_ull, assumed;

  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN
    // (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

template<typename T> __forceinline__ __device__ auto atomicAdd_block(T *address, T val) {
  return atomicAdd(address, val);
}

#endif

/**
 * It computes the start and end point of the spreading window given the center x and the
 * width ns.
 * TODO: We should move to (md)spans and (nd)ranges to avoid xend.
 *       It is also safer on bounds.
 */
template<typename T> __forceinline__ __device__ auto interval(const int ns, const T x) {
  const auto xstart = int(std::ceil(x - T(ns) * T(.5)));
  const auto xend   = xstart + ns - 1;
  return int2{xstart, xend};
}

#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ >= 900
#define COMPUTE_CAPABILITY_90_OR_HIGHER 1
#else
#define COMPUTE_CAPABILITY_90_OR_HIGHER 0
#endif
#else
#define COMPUTE_CAPABILITY_90_OR_HIGHER 0
#endif

namespace cufinufft {
namespace utils {

using namespace finufft::common;

/**
 * does a complex atomic add on a shared memory address
 * it adds the real and imaginary parts separately
 * cuda does not support atomic operations
 * on complex numbers on shared memory directly
 */
template<typename T>
static __forceinline__ __device__ void atomicAddComplexShared(
    cuda_complex<T> *address, const cuda_complex<T> &res) {
  const auto raw_address = reinterpret_cast<T *>(address);
  atomicAdd_block(raw_address, res.x);
  atomicAdd_block(raw_address + 1, res.y);
}

/**
 * does a complex atomic add on a global memory address
 * since cuda 90 atomic operations on complex numbers
 * on shared memory are supported so we leverage them
 */
template<typename T>
static __forceinline__ __device__ void atomicAddComplexGlobal(cuda_complex<T> *address,
                                                              cuda_complex<T> res) {
  if constexpr (
      std::is_same_v<cuda_complex<T>, float2> && COMPUTE_CAPABILITY_90_OR_HIGHER) {
    atomicAdd(address, res);
  } else {
    auto raw_address = reinterpret_cast<T *>(address);
    atomicAdd(raw_address, res.x);
    atomicAdd(raw_address + 1, res.y);
  }
}

template<typename T> auto arrayrange(int n, const T *a, cudaStream_t stream) {
  const auto d_min_max = thrust::minmax_element(thrust::cuda::par.on(stream), a, a + n);

  // copy d_min and d_max to host
  T min{}, max{};
  checkCudaErrors(cudaMemcpyAsync(&min, thrust::raw_pointer_cast(d_min_max.first),
                                  sizeof(T), cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaMemcpyAsync(&max, thrust::raw_pointer_cast(d_min_max.second),
                                  sizeof(T), cudaMemcpyDeviceToHost, stream));
  return std::make_tuple(min, max);
}

// Writes out w = half-width and c = center of an interval enclosing all a[n]'s
// Only chooses a nonzero center if this increases w by less than fraction
// ARRAYWIDCEN_GROWFRAC defined in finufft_common/constants.h.
// This prevents rephasings which don't grow nf by much. 6/8/17
// If n==0, w and c are not finite.
template<typename T> auto arraywidcen(int n, const T *a, cudaStream_t stream) {
  const auto [lo, hi] = arrayrange(n, a, stream);
  auto w              = (hi - lo) / 2;
  auto c              = (hi + lo) / 2;
  if (std::abs(c) < ARRAYWIDCEN_GROWFRAC * w) {
    w += std::abs(c);
    c = 0.0;
  }
  return std::make_tuple(w, c);
}

/* Compile-time dispatch on the kernel width ns, the one shape parameter every
   spread/interp kernel templates on; the Horner row count follows from ns
   (finufft::kernel::max_nc_given_ns<T>). The caller is taken by value because
   poet::dispatch needs a mutable functor. Caller exposes
   template<int Ns> void operator()() const. */
template<typename T, class Caller> void dispatch_kernel_shape(Caller caller, int ns) {
  using NsSeq = poet::inclusive_range<finufft::common::MIN_NSPREAD,
                                      finufft::common::MAX_NSPREAD<T>>;
  poet::dispatch(caller, std::make_tuple(poet::dispatch_param<NsSeq>{ns}));
}

} // namespace utils
} // namespace cufinufft

#endif
