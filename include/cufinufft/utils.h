#ifndef FINUFFT_UTILS_H
#define FINUFFT_UTILS_H

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <complex>
#include <cstdint>

#include <cuComplex.h>
#include <cufinufft/types.h>

#include <cuda_runtime.h>

#include <sys/time.h>

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
  WithCudaDevice(int device) {
    cudaGetDevice(&orig_device_);
    cudaSetDevice(device);
  }

  ~WithCudaDevice() { cudaSetDevice(orig_device_); }

private:
  int orig_device_;
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
} // namespace utils
} // namespace cufinufft

#endif
