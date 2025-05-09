#pragma once

#ifdef __NVCC__
#include <cuda_runtime.h>
#endif

namespace cufinufft {
/**
 * @brief Load a value from global memory using the read-only data cache.
 *
 * Uses __ldg() to load data through the read-only data cache. Optimized for
 * values that are not modified during kernel execution and may be accessed by
 * multiple threads, improving cache efficiency and reducing memory latency.
 */
template<typename T> __device__ __forceinline__ T loadReadOnly(const T *ptr) {
#ifdef __CUDA_ARCH__
  return __ldg(ptr);
#else
  return *ptr;
#endif
}

/**
 * @brief Load a value with caching at all levels (L1 and L2).
 *
 * Uses __ldca() to ensure data is cached in both L1 and L2. Ideal for data
 * accessed repeatedly by threads in a block or across blocks.
 */
template<typename T> __device__ __forceinline__ T loadCacheAll(const T *ptr) {
#ifdef __NVCC__
  return __ldca(ptr);
#else
  return *ptr;
#endif
}

/**
 * @brief Load a value with caching at the global level (L2), bypassing L1.
 *
 * Uses __ldcg() for loads that bypass L1 and go through L2 only. Suitable
 * when you want to reduce L1 cache pollution.
 */
template<typename T> __device__ __forceinline__ T loadCacheGlobal(const T *ptr) {
#ifdef __NVCC__
  return __ldcg(ptr);
#else
  return *ptr;
#endif
}

/**
 * @brief Load a value with a streaming cache policy.
 *
 * Uses __ldcs() for data that is expected to be accessed only once.
 * Optimizes for throughput without polluting caches.
 */
template<typename T> __device__ __forceinline__ T loadCacheStreaming(const T *ptr) {
#ifdef __NVCC__
  return __ldcs(ptr);
#else
  return *ptr;
#endif
}

/**
 * @brief Load a value with volatile semantics.
 *
 * Uses __ldcv() to load volatile data. Ensures that every read sees the
 * most recent write.
 */
template<typename T> __device__ __forceinline__ T loadCacheVolatile(const T *ptr) {
#ifdef __NVCC__
  return __ldcv(ptr);
#else
  return *ptr;
#endif
}

/**
 * @brief Load a value with last-use semantics.
 *
 * Uses __ldlu() to hint that the value will not be reused. May improve
 * eviction performance in certain access patterns.
 */
template<typename T> __device__ __forceinline__ T loadLastUse(const T *ptr) {
#ifdef __NVCC__
  return __ldlu(ptr);
#else
  return *ptr;
#endif
}

/**
 * @brief Store a value with write-back policy.
 *
 * Uses __stwb() to write data into memory via cache. The store may be
 * buffered and deferred.
 */
template<typename T> __device__ __forceinline__ void storeWriteBack(T *ptr, T value) {
#ifdef __NVCC__
  __stwb(ptr, value);
#else
  *ptr = value;
#endif
}

/**
 * @brief Store a value caching only in the global level (L2), bypassing L1.
 *
 * Uses __stcg() to reduce L1 pollution for data not expected to be reused.
 */
template<typename T> __device__ __forceinline__ void storeCacheGlobal(T *ptr, T value) {
#ifdef __NVCC__
  __stcg(ptr, value);
#else
  *ptr = value;
#endif
}

/**
 * @brief Store a value with streaming policy.
 *
 * Uses __stcs() to write data with a hint that it's accessed in a
 * streaming fashion (written once, not reused).
 */
template<typename T>
__device__ __forceinline__ void storeCacheStreaming(T *ptr, T value) {
#ifdef __NVCC__
  __stcs(ptr, value);
#else
  *ptr = value;
#endif
}

/**
 * @brief Store a value using write-through policy.
 *
 * Uses __stwt() to ensure the data is immediately written to memory, bypassing cache.
 */
template<typename T> __device__ __forceinline__ void storeWriteThrough(T *ptr, T value) {
#ifdef __NVCC__
  __stwt(ptr, value);
#else
  *ptr = value;
#endif
}
} // namespace cufinufft
