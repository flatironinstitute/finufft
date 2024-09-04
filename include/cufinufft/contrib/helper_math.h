#ifndef FINUFFT_INCLUDE_CUFINUFFT_CONTRIB_HELPER_MATH_H
#define FINUFFT_INCLUDE_CUFINUFFT_CONTRIB_HELPER_MATH_H

#include <cuComplex.h>

// This header provides some helper functions for cuComplex types.
// It mainly wraps existing CUDA implementations to provide operator overloads
// e.g. cuAdd, cuSub, cuMul, cuDiv, cuCreal, cuCimag, cuCabs, cuCarg, cuConj are all
// provided by CUDA

// Addition for cuDoubleComplex (double) with cuDoubleComplex (double)
__host__ __device__ __forceinline__ cuDoubleComplex operator+(
    const cuDoubleComplex &a, const cuDoubleComplex &b) noexcept {
  return cuCadd(a, b);
}

// Subtraction for cuDoubleComplex (double) with cuDoubleComplex (double)
__host__ __device__ __forceinline__ cuDoubleComplex operator-(
    const cuDoubleComplex &a, const cuDoubleComplex &b) noexcept {
  return cuCsub(a, b);
}

// Multiplication for cuDoubleComplex (double) with cuDoubleComplex (double)
__host__ __device__ __forceinline__ cuDoubleComplex operator*(
    const cuDoubleComplex &a, const cuDoubleComplex &b) noexcept {
  return cuCmul(a, b);
}

// Division for cuDoubleComplex (double) with cuDoubleComplex (double)
__host__ __device__ __forceinline__ cuDoubleComplex operator/(
    const cuDoubleComplex &a, const cuDoubleComplex &b) noexcept {
  return cuCdiv(a, b);
}

// Equality for cuDoubleComplex (double) with cuDoubleComplex (double)
__host__ __device__ __forceinline__ bool operator==(const cuDoubleComplex &a,
                                                    const cuDoubleComplex &b) noexcept {
  return cuCreal(a) == cuCreal(b) && cuCimag(a) == cuCimag(b);
}

// Inequality for cuDoubleComplex (double) with cuDoubleComplex (double)
__host__ __device__ __forceinline__ bool operator!=(const cuDoubleComplex &a,
                                                    const cuDoubleComplex &b) noexcept {
  return !(a == b);
}

// Addition for cuDoubleComplex (double) with double
__host__ __device__ __forceinline__ cuDoubleComplex operator+(const cuDoubleComplex &a,
                                                              double b) noexcept {
  return make_cuDoubleComplex(cuCreal(a) + b, cuCimag(a));
}

__host__ __device__ __forceinline__ cuDoubleComplex operator+(
    double a, const cuDoubleComplex &b) noexcept {
  return make_cuDoubleComplex(a + cuCreal(b), cuCimag(b));
}

// Subtraction for cuDoubleComplex (double) with double
__host__ __device__ __forceinline__ cuDoubleComplex operator-(const cuDoubleComplex &a,
                                                              double b) noexcept {
  return make_cuDoubleComplex(cuCreal(a) - b, cuCimag(a));
}

__host__ __device__ __forceinline__ cuDoubleComplex operator-(
    double a, const cuDoubleComplex &b) noexcept {
  return make_cuDoubleComplex(a - cuCreal(b), -cuCimag(b));
}

// Multiplication for cuDoubleComplex (double) with double
__host__ __device__ __forceinline__ cuDoubleComplex operator*(const cuDoubleComplex &a,
                                                              double b) noexcept {
  return make_cuDoubleComplex(cuCreal(a) * b, cuCimag(a) * b);
}

__host__ __device__ __forceinline__ cuDoubleComplex operator*(
    double a, const cuDoubleComplex &b) noexcept {
  return make_cuDoubleComplex(a * cuCreal(b), a * cuCimag(b));
}

// Division for cuDoubleComplex (double) with double
__host__ __device__ __forceinline__ cuDoubleComplex operator/(const cuDoubleComplex &a,
                                                              double b) noexcept {
  return make_cuDoubleComplex(cuCreal(a) / b, cuCimag(a) / b);
}

__host__ __device__ __forceinline__ cuDoubleComplex operator/(
    double a, const cuDoubleComplex &b) noexcept {
  double denom = cuCreal(b) * cuCreal(b) + cuCimag(b) * cuCimag(b);
  return make_cuDoubleComplex((a * cuCreal(b)) / denom, (-a * cuCimag(b)) / denom);
}

// Addition for cuFloatComplex (float) with cuFloatComplex (float)
__host__ __device__ __forceinline__ cuFloatComplex operator+(
    const cuFloatComplex &a, const cuFloatComplex &b) noexcept {
  return cuCaddf(a, b);
}

// Subtraction for cuFloatComplex (float) with cuFloatComplex (float)
__host__ __device__ __forceinline__ cuFloatComplex operator-(
    const cuFloatComplex &a, const cuFloatComplex &b) noexcept {
  return cuCsubf(a, b);
}

// Multiplication for cuFloatComplex (float) with cuFloatComplex (float)
__host__ __device__ __forceinline__ cuFloatComplex operator*(
    const cuFloatComplex &a, const cuFloatComplex &b) noexcept {
  return cuCmulf(a, b);
}

// Division for cuFloatComplex (float) with cuFloatComplex (float)
__host__ __device__ __forceinline__ cuFloatComplex operator/(
    const cuFloatComplex &a, const cuFloatComplex &b) noexcept {
  return cuCdivf(a, b);
}

// Equality for cuFloatComplex (float) with cuFloatComplex (float)
__host__ __device__ __forceinline__ bool operator==(const cuFloatComplex &a,
                                                    const cuFloatComplex &b) noexcept {
  return cuCrealf(a) == cuCrealf(b) && cuCimagf(a) == cuCimagf(b);
}

// Inequality for cuFloatComplex (float) with cuFloatComplex (float)
__host__ __device__ __forceinline__ bool operator!=(const cuFloatComplex &a,
                                                    const cuFloatComplex &b) noexcept {
  return !(a == b);
}

// Addition for cuFloatComplex (float) with float
__host__ __device__ __forceinline__ cuFloatComplex operator+(const cuFloatComplex &a,
                                                             float b) noexcept {
  return make_cuFloatComplex(cuCrealf(a) + b, cuCimagf(a));
}

__host__ __device__ __forceinline__ cuFloatComplex operator+(
    float a, const cuFloatComplex &b) noexcept {
  return make_cuFloatComplex(a + cuCrealf(b), cuCimagf(b));
}

// Subtraction for cuFloatComplex (float) with float
__host__ __device__ __forceinline__ cuFloatComplex operator-(const cuFloatComplex &a,
                                                             float b) noexcept {
  return make_cuFloatComplex(cuCrealf(a) - b, cuCimagf(a));
}

__host__ __device__ __forceinline__ cuFloatComplex operator-(
    float a, const cuFloatComplex &b) noexcept {
  return make_cuFloatComplex(a - cuCrealf(b), -cuCimagf(b));
}

// Multiplication for cuFloatComplex (float) with float
__host__ __device__ __forceinline__ cuFloatComplex operator*(const cuFloatComplex &a,
                                                             float b) noexcept {
  return make_cuFloatComplex(cuCrealf(a) * b, cuCimagf(a) * b);
}

__host__ __device__ __forceinline__ cuFloatComplex operator*(
    float a, const cuFloatComplex &b) noexcept {
  return make_cuFloatComplex(a * cuCrealf(b), a * cuCimagf(b));
}

// Division for cuFloatComplex (float) with float
__host__ __device__ __forceinline__ cuFloatComplex operator/(const cuFloatComplex &a,
                                                             float b) noexcept {
  return make_cuFloatComplex(cuCrealf(a) / b, cuCimagf(a) / b);
}

__host__ __device__ __forceinline__ cuFloatComplex operator/(
    float a, const cuFloatComplex &b) noexcept {
  float denom = cuCrealf(b) * cuCrealf(b) + cuCimagf(b) * cuCimagf(b);
  return make_cuFloatComplex((a * cuCrealf(b)) / denom, (-a * cuCimagf(b)) / denom);
}

#endif // FINUFFT_INCLUDE_CUFINUFFT_CONTRIB_HELPER_MATH_H
