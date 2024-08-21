#ifndef FINUFFT_UTILS_H
#define FINUFFT_UTILS_H

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <complex>
#include <cstdint>

#include <cuComplex.h>
#include <cufinufft/types.h>

#include <cuda_runtime.h>

#include <sys/time.h>

#include <cuda.h>
#include <type_traits>

#include <thrust/extrema.h>

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

// jfm timer class
class CNTime {
public:
  void start();
  double restart();
  double elapsedsec();

private:
  struct timeval initial;
};

// ahb math helpers
CUFINUFFT_BIGINT next235beven(CUFINUFFT_BIGINT n, CUFINUFFT_BIGINT b);

template<typename T> T infnorm(int n, std::complex<T> *a) {
  T nrm = 0.0;
  for (int m = 0; m < n; ++m) {
    T aa = real(conj(a[m]) * a[m]);
    if (aa > nrm) nrm = aa;
  }
  return sqrt(nrm);
}

#ifdef __CUDA_ARCH__
__forceinline__ __device__ auto interval(const int ns, const float x) {
  // float to int round up and fused multiply-add to round up
  const auto xstart = __float2int_ru(__fmaf_ru(ns, -.5f, x));
  // float to int round down and fused multiply-add to round down
  const auto xend = __float2int_rd(__fmaf_rd(ns, .5f, x));
  return int2{xstart, xend};
}
__forceinline__ __device__ auto interval(const int ns, const double x) {
  // same as above
  const auto xstart = __double2int_ru(__fma_ru(ns, -.5, x));
  const auto xend   = __double2int_rd(__fma_rd(ns, .5, x));
  return int2{xstart, xend};
}
#endif

// Define a macro to check if NVCC version is >= 11.3
#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__)
#if (__CUDACC_VER_MAJOR__ > 11) || \
    (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 3 && __CUDA_ARCH__ >= 600)
#define ALLOCA_SUPPORTED 1
// windows compatibility
#if __has_include(<malloc.h>)
#include <malloc.h>
#endif
#endif
#endif

#undef ALLOCA_SUPPORTED

#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ >= 900
#define COMPUTE_CAPABILITY_90_OR_HIGHER 1
#else
#define COMPUTE_CAPABILITY_90_OR_HIGHER 0
#endif
#else
#define COMPUTE_CAPABILITY_90_OR_HIGHER 0
#endif

/**
 * does a complex atomic add on a shared memory address
 * it adds the real and imaginary parts separately
 * cuda does not support atomic operations
 * on complex numbers on shared memory directly
 */

template<typename T>
static __forceinline__ __device__ void atomicAddComplexShared(
    cuda_complex<T> *address, cuda_complex<T> res) {
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
static __forceinline__ __device__ void atomicAddComplexGlobal(
    cuda_complex<T> *address, cuda_complex<T> res) {
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
  checkCudaErrors(cudaMemcpy(&min, thrust::raw_pointer_cast(d_min_max.first), sizeof(T),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&max, thrust::raw_pointer_cast(d_min_max.second), sizeof(T),
                             cudaMemcpyDeviceToHost));
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
/* sets nf, h (upsampled grid spacing), and gamma (x_j rescaling factor),
   for type 3 only.
   Inputs:
   X and S are the xj and sk interval half-widths respectively.
   opts and spopts are the NUFFT and spreader opts strucs, respectively.
   Outputs:
   nf is the size of upsampled grid for a given single dimension.
   h is the grid spacing = 2pi/nf
   gam is the x rescale factor, ie x'_j = x_j/gam  (modulo shifts).
   Barnett 2/13/17. Caught inf/nan 3/14/17. io int types changed 3/28/17
   New logic 6/12/17
*/
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

} // namespace utils
} // namespace cufinufft

#endif
