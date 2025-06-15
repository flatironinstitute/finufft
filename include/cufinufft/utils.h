#ifndef FINUFFT_UTILS_H
#define FINUFFT_UTILS_H

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <complex>

#include <cuComplex.h>
#include <cufinufft/types.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <utility> // for std::forward

#include <thrust/extrema.h>

#include <finufft_errors.h>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
#endif

/**
 * It computes the stard and end point of the spreading window given the center x and the
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

// math helpers whose source is in src/cuda/utils.cpp
CUFINUFFT_BIGINT next235beven(CUFINUFFT_BIGINT n, CUFINUFFT_BIGINT b);
void gaussquad(int n, double *xgl, double *wgl);
std::tuple<double, double> leg_eval(int n, double x);

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
static __forceinline__ __device__ void atomicAddComplexShared(cuda_complex<T> *address,
                                                              cuda_complex<T> res) {
  const auto raw_address = reinterpret_cast<T *>(address);
  atomicAdd(raw_address, res.x);
  atomicAdd(raw_address + 1, res.y);
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
    atomicAddComplexShared<T>(address, res);
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
// ARRAYWIDCEN_GROWFRAC defined in defs.h.
// This prevents rephasings which don't grow nf by much. 6/8/17
// If n==0, w and c are not finite.
template<typename T> auto arraywidcen(int n, T *a, cudaStream_t stream) {
  const auto [lo, hi] = arrayrange(n, a, stream);
  auto w              = (hi - lo) / 2;
  auto c              = (hi + lo) / 2;
  if (std::abs(c) < ARRAYWIDCEN_GROWFRAC * w) {
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
  T nfd = 2.0 * opts.upsampfac * Ssafe * Xsafe / M_PI + nss;
  if (!std::isfinite(nfd)) nfd = 0.0; // use FLT to catch inf
  auto nf = (int)nfd;
  // printf("initial nf=%lld, ns=%d\n",*nf,spopts.nspread);
  //  catch too small nf, and nan or +-inf, otherwise spread fails...
  if (nf < 2 * spopts.nspread) nf = 2 * spopts.nspread;
  if (nf < MAX_NF)                   // otherwise will fail anyway
    nf = utils::next235beven(nf, 1); // expensive at huge nf
  // Note: b is 1 because type 3 uses a type 2 plan, so it should not need the extra
  // condition that seems to be used by Block Gather as type 2 are only GM-sort
  auto h   = 2 * T(M_PI) / nf;                       // upsampled grid spacing
  auto gam = T(nf) / (2.0 * opts.upsampfac * Ssafe); // x scale fac to x'
  return std::make_tuple(nf, h, gam);
}

// Generalized dispatcher for any function requiring ns-based dispatch
template<typename Func, typename T, int ns, typename... Args>
int dispatch_ns(Func &&func, int target_ns, Args &&...args) {
  if constexpr (ns > MAX_NSPREAD) {
    return FINUFFT_ERR_METHOD_NOTVALID; // Stop recursion
  } else {
    if (target_ns == ns) {
      return std::forward<Func>(func).template operator()<ns>(
          std::forward<Args>(args)...);
    }
    return dispatch_ns<Func, T, ns + 1>(std::forward<Func>(func), target_ns,
                                        std::forward<Args>(args)...);
  }
}

// Wrapper function that starts the dispatch recursion
template<typename Func, typename T, typename... Args>
int launch_dispatch_ns(Func &&func, int target_ns, Args &&...args) {
  return dispatch_ns<Func, T, MIN_NSPREAD>(std::forward<Func>(func), target_ns,
                                           std::forward<Args>(args)...);
}

} // namespace utils
} // namespace cufinufft

#endif
