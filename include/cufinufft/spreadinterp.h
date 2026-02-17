#ifndef __CUSPREADINTERP_H__
#define __CUSPREADINTERP_H__

#include <cmath>
#include <cuda.h>
#include <cufinufft/types.h>
#include <finufft_common/spread_opts.h>

namespace cufinufft {
namespace spreadinterp {

template<typename T>
static constexpr __forceinline__ __device__ T cudaFMA(const T a, const T b, const T c) {
  if constexpr (std::is_same_v<T, float>) {
    // fused multiply-add, round to nearest even
    return __fmaf_rn(a, b, c);
  } else if constexpr (std::is_same_v<T, double>) {
    // fused multiply-add, round to nearest even
    return __fma_rn(a, b, c);
  }
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "Only float and double are supported.");
  return std::fma(a, b, c);
}

/**
 * local NU coord fold+rescale macro: does the following affine transform to x:
 *   (x+PI) mod PI    each to [0,N)
 */
template<typename T>
constexpr __forceinline__ __host__ __device__ T fold_rescale(T x, int N) {
  constexpr auto x2pi = T(0.159154943091895345554011992339482617);
  constexpr auto half = T(0.5);
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<T, float>) {
    // fused multiply-add, round to nearest even
    auto result = cudaFMA(x, x2pi, half);
    // subtract, round down
    result = __fsub_rd(result, floorf(result));
    // multiply, round down
    return __fmul_rd(result, static_cast<T>(N));
  } else if constexpr (std::is_same_v<T, double>) {
    // fused multiply-add, round to nearest even
    auto result = cudaFMA(x, x2pi, half);
    // subtract, round down
    result = __dsub_rd(result, floor(result));
    // multiply, round down
    return __dmul_rd(result, static_cast<T>(N));
  } else {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "Only float and double are supported.");
  }
#else
  const auto result = std::fma(x, x2pi, half);
  return (result - std::floor(result)) * static_cast<T>(N);
#endif
}

template<typename T>
int setup_spreader(finufft_spread_opts &spopts, T eps, T upsampfac, int kerevalmeth,
                   int debug, int spreadinterponly);

template<typename T>
static inline T evaluate_kernel(T x, const finufft_spread_opts &spopts)
/* ES ("exp sqrt" or "exp semicircle") kernel evaluation, single real argument:
   returns phi(2x/ns) := exp(beta.[sqrt(1 - (2x/ns)^2) - 1]),  for |x| < ns/2.
   This is the reference implementation, used by src/cuda/common.cu for onedim
   FT quadrature approx, so it need not be fast.
   This is the original kernel used 2017-2025 in CPU code, as in [FIN] paper.
   This is related to an asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   To do: *** replace kernel logic/coeffs with CPU codes in common/
*/
{
  T z = 2.0 * x / T(spopts.nspread); // argument on [-1,1]
  if (abs(z) >= 1.0)
    // if spreading/FT careful, shouldn't need this if, but causes no speed hit
    return 0.0;
  else {
    return exp((T)spopts.beta * (sqrt((T)1.0 - z * z) - (T)1.0));
  }
}

template<typename T, int ns>
static __forceinline__ __device__ T evaluate_kernel(T x, T es_c, T es_beta)
// ns (spread-width) templated fast evaluator for the above kernel function.
// Direct eval, hardwired for this kernel, with shape param es_beta.
// es_c must have been set up as (2/ns)^2; the point of precomputing this is to
// reduce flops (as in the original CPU kernel). es_halfwidth has been cut.
// Used only by the below eval_kernel_vec(), hence when gpu_kerevalmeth=0.
// To do: *** unify kernel logic/coeffs with CPU codes in common/
{
  const T zsq = es_c * x * x; // z^2, where z is arg for std interval [-1,1]
  return (zsq < 1.0) ? exp((T)es_beta * (sqrt((T)1.0 - zsq) - (T)1.0)) : 0.0;
}

template<typename T, int w>
static __inline__ __device__ void eval_kernel_vec(T *ker, const T x, const T es_c,
                                                  const T es_beta) {
  // Eval the above direct ES kernel evaluator for arguments x+j, for j=0,..,w-1.
  // This is used when gpu_kerevalmeth=0.
  // Serves the same purpose as the below function eval_kernel_vec_horner.
  for (int i = 0; i < w; i++)
    ker[i] = evaluate_kernel<T, w>(abs(x + i), es_c, es_beta);
}

template<typename T, int w>
static __device__ void eval_kernel_vec_horner(T *ker, const T x, const double upsampfac)
/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
   x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
   Two upsampfacs implemented (same as CPU coeffs created in 2018, used to 2025).
   The parameter (w and beta) choice in setup_spreader must match these coeffs.
   This is used when gpu_kerevalmeth=1.
   To do: *** update horner evaluator and coeffs to match CPU in common/
   */
{
  // const T z = T(2) * x + T(w - 1);
  const auto z = cudaFMA(T(2), x, T(w - 1)); // scale so local grid offset z in [-1,1]
  // insert the auto-generated code which expects z, w args, writes to ker...
  if (upsampfac == 2.0) { // floating point equality is fine here
    using FLT = T;
#include "cufinufft/contrib/ker_horner_allw_loop.inc"
  }
  if (upsampfac == 1.25) { // floating point equality is fine here
    using FLT = T;
#include "cufinufft/contrib/ker_lowupsampfac_horner_allw_loop.inc"
  }
}

// Functions for calling different methods of spreading & interpolation
template<typename T> int cuspread1d(cufinufft_plan_t<T> *d_plan, int blksize);
template<typename T> int cuinterp1d(cufinufft_plan_t<T> *d_plan, int blksize);

template<typename T> int cuspread2d(cufinufft_plan_t<T> *d_plan, int blksize);
template<typename T> int cuinterp2d(cufinufft_plan_t<T> *d_plan, int blksize);

template<typename T> int cuspread3d(cufinufft_plan_t<T> *d_plan, int blksize);
template<typename T> int cuinterp3d(cufinufft_plan_t<T> *d_plan, int blksize);

// Wrappers for methods of spreading
template<typename T>
int cuspread1d_nuptsdriven_prop(int nf1, int M, cufinufft_plan_t<T> *d_plan);
template<typename T, int ns>
int cuspread1d_nuptsdriven(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize);
template<typename T>
int cuspread1d_subprob_prop(int nf1, int M, cufinufft_plan_t<T> *d_plan);
template<typename T, int ns>
int cuspread1d_subprob(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize);
template<typename T, int ns>
int cuspread1d_output_driven(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize);
template<typename T>
int cuspread2d_nuptsdriven_prop(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan);
template<typename T, int ns>
int cuspread2d_nuptsdriven(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan,
                           int blksize);
template<typename T>
int cuspread2d_subprob_prop(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan);
template<typename T, int ns>
int cuspread2d_subprob(int nf1, int nf2, int m, cufinufft_plan_t<T> *d_plan, int blksize);
template<typename T, int ns>
int cuspread2d_output_driven(int nf1, int nf2, int m, cufinufft_plan_t<T> *d_plan,
                             int blksize);
template<typename T>
int cuspread3d_nuptsdriven_prop(int nf1, int nf2, int nf3, int M,
                                cufinufft_plan_t<T> *d_plan);
template<typename T, int ns>
int cuspread3d_nuptsdriven(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
                           int blksize);
template<typename T>
int cuspread3d_blockgather_prop(int nf1, int nf2, int nf3, int M,
                                cufinufft_plan_t<T> *d_plan);
template<typename T, int ns>
int cuspread3d_blockgather(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
                           int blksize);
template<typename T>
int cuspread3d_subprob_prop(int nf1, int nf2, int nf3, int M,
                            cufinufft_plan_t<T> *d_plan);
template<typename T, int ns>
int cuspread3d_subprob(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
                       int blksize);

template<typename T, int ns>
int cuspread3d_output_driven(int nf1, int nf2, int nf3, int M,
                             cufinufft_plan_t<T> *d_plan, int blksize);

// Wrappers for methods of interpolation
template<typename T, int ns>
int cuinterp2d_nuptsdriven(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan,
                           int blksize);
template<typename T, int ns>
int cuinterp2d_subprob(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan, int blksize);
template<typename T, int ns>
int cuinterp3d_nuptsdriven(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
                           int blksize);
template<typename T, int ns>
int cuinterp3d_subprob(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
                       int blksize);

} // namespace spreadinterp
} // namespace cufinufft
#endif
