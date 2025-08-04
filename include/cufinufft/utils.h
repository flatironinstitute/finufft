#ifndef FINUFFT_UTILS_H
#define FINUFFT_UTILS_H

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <complex>

#include <cuComplex.h>
#include <cufinufft/types.h>

#include <array>
#include <cuda_runtime.h>
#include <thrust/extrema.h>
#include <tuple>
#include <type_traits>
#include <utility> // for std::forward

#include <common/common.h>
#include <finufft_errors.h>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>

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

class WithCudaDevice {
public:
  explicit WithCudaDevice(const int device) : orig_device_{get_orig_device()} {
    cudaSetDevice(device);
  }

  ~WithCudaDevice() { cudaSetDevice(orig_device_); }

private:
  const int orig_device_;

  static int get_orig_device() noexcept {
    int device{};
    cudaGetDevice(&device);
    return device;
  }
};

// math helpers whose source is in src/utils.cpp
long next235beven(long n, long b);

template<typename T> T infnorm(int n, std::complex<T> *a) {
  T nrm = 0.0;
  for (int m = 0; m < n; ++m) {
    T aa = real(conj(a[m]) * a[m]);
    if (aa > nrm) nrm = aa;
  }
  return sqrt(nrm);
}

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

template<typename T> auto arrayrange(int n, T *a, cudaStream_t stream) {
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
// ARRAYWIDCEN_GROWFRAC defined in common/constants.h.
// This prevents rephasings which don't grow nf by much. 6/8/17
// If n==0, w and c are not finite.
template<typename T> auto arraywidcen(int n, T *a, cudaStream_t stream) {
  const auto [lo, hi] = arrayrange(n, a, stream);
  auto w              = (hi - lo) / 2;
  auto c              = (hi + lo) / 2;
  if (std::abs(c) < ::finufft::common::ARRAYWIDCEN_GROWFRAC * w) {
    w += std::abs(c);
    c = 0.0;
  }
  return std::make_tuple(w, c);
}

template<typename T>
auto set_nhg_type3(T S, T X, const cufinufft_opts &opts,
                   const finufft_spread_opts &spopts)
// It implements the same named function in finufft_core.cpp (see its docs)
{
  int nss = spopts.nspread + 1; // since ns may be odd
  T Xsafe = X, Ssafe = S;       // may be tweaked locally
  if (X == 0.0)                 // logic ensures XS>=1, handle X=0 a/o S=0
    if (S == 0.0) {
      Xsafe = 1.0;
      Ssafe = 1.0;
    } else
      Xsafe = std::max(Xsafe, T(1) / S);
  else
    Ssafe = std::max(Ssafe, T(1) / X);
  // use the safe X and S...
  T nfd = 2.0 * opts.upsampfac * Ssafe * Xsafe / finufft::common::PI + nss;
  if (!std::isfinite(nfd)) nfd = 0.0; // use FLT to catch inf
  auto nf = (int)nfd;
  // printf("initial nf=%lld, ns=%d\n",*nf,spopts.nspread);
  //  catch too small nf, and nan or +-inf, otherwise spread fails...
  if (nf < 2 * spopts.nspread) nf = 2 * spopts.nspread;
  if (nf < MAX_NF)                   // otherwise will fail anyway
    nf = utils::next235beven(nf, 1); // expensive at huge nf
  // Note: b is 1 because type 3 uses a type 2 plan, so it should not need the extra
  // condition that seems to be used by Block Gather as type 2 are only GM-sort
  auto h   = 2 * T(finufft::common::PI) / nf;        // upsampled grid spacing
  auto gam = T(nf) / (2.0 * opts.upsampfac * Ssafe); // x scale fac to x'
  return std::make_tuple(nf, h, gam);
}

// Wrapper around the generic dispatcher for nspread-based dispatch
template<typename Func, typename T, typename... Args>
auto launch_dispatch_ns(Func &&func, int target_ns, Args &&...args) {
  using NsSeq = finufft::common::make_range<::finufft::common::MIN_NSPREAD,
                                            ::finufft::common::MAX_NSPREAD>;
  auto params =
      std::make_tuple(finufft::common::DispatchParam<NsSeq>{target_ns, NsSeq{}});
  return finufft::common::dispatch(std::forward<Func>(func), params,
                                   std::forward<Args>(args)...);
}

/**
 * Return an architecture-specific “good enough” thread-block size.
 * – Each branch is resolved at compile time (if-constexpr + __CUDA_ARCH__).
 * – Host-only translation units get the fall-back value.
 * Rationale (rule-of-thumb):
 *   SM 9x / 8x : 16 warps  = 256 threads
 *   SM 7x      :  8 warps  = 128 threads
 *   SM 6x-     :  4 warps  = 64 threads
 */
inline unsigned optimal_block_threads(int device) noexcept {
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  int major = prop.major;
  if (major >= 8) return 256;
  if (major >= 7) return 128;
  return 64;
}

} // namespace utils
} // namespace cufinufft

#endif
