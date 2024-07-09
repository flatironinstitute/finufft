#ifndef __CUSPREADINTERP_H__
#define __CUSPREADINTERP_H__

#include <cmath>
#include <cuda.h>
#include <cufinufft/types.h>
#include <finufft_spread_opts.h>

namespace cufinufft {
namespace spreadinterp {

template<typename T> static __forceinline__ __device__ T fold_rescale(T x, int N) {
  static constexpr const auto x2pi = T(0.159154943091895345554011992339482617);
  const T result                   = x * x2pi + T(0.5);
  return (result - floor(result)) * T(N);
}

template<typename T>
static inline T evaluate_kernel(T x, const finufft_spread_opts &opts)
/* ES ("exp sqrt") kernel evaluation at single real argument:
      phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2
   related to an asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   This is the "reference implementation", used by eg common/onedim_* 2/17/17 */
{
  if (abs(x) >= opts.ES_halfwidth)
    // if spreading/FT careful, shouldn't need this if, but causes no speed hit
    return 0.0;
  else
    return exp(opts.ES_beta * sqrt(1.0 - opts.ES_c * x * x));
}

template<typename T>
int setup_spreader(finufft_spread_opts &opts, T eps, T upsampfac, int kerevalmeth);

template<typename T>
static __forceinline__ __device__ T evaluate_kernel(T x, T es_c, T es_beta, int ns)
/* ES ("exp sqrt") kernel evaluation at single real argument:
   phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2
   related to an asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   This is the "reference implementation", used by eg common/onedim_*
    2/17/17 */
{
  return abs(x) < ns / 2.0 ? exp(es_beta * (sqrt(1.0 - es_c * x * x))) : 0.0;
}

template<typename T>
static __inline__ __device__ void eval_kernel_vec_horner(T *ker, const T x, const int w,
                                                         const double upsampfac)
/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
   x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
   This is the current evaluation method, since it's faster (except i7 w=16).
   Two upsampfacs implemented. Params must match ref formula. Barnett 4/24/18 */
{
  T z = 2 * x + w - 1.0; // scale so local grid offset z in [-1,1]
  // insert the auto-generated code which expects z, w args, writes to ker...
  if (upsampfac == 2.0) { // floating point equality is fine here
    using FLT           = T;
    using CUFINUFFT_FLT = T;
#include "cufinufft/contrib/ker_horner_allw_loop.inc"
  }
}

template<typename T>
static __inline__ __device__ void eval_kernel_vec(T *ker, const T x, const int w,
                                                  const T es_c, const T es_beta) {
  for (int i = 0; i < w; i++) {
    ker[i] = evaluate_kernel(abs(x + i), es_c, es_beta, w);
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
template<typename T>
int cuspread1d_nuptsdriven(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize);
template<typename T>
int cuspread1d_subprob_prop(int nf1, int M, cufinufft_plan_t<T> *d_plan);
template<typename T>
int cuspread1d_subprob(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize);

template<typename T>
int cuspread2d_nuptsdriven_prop(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan);
template<typename T>
int cuspread2d_nuptsdriven(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan,
                           int blksize);
template<typename T>
int cuspread2d_subprob_prop(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan);
template<typename T>
int cuspread2d_subprob(int nf1, int nf2, int m, cufinufft_plan_t<T> *d_plan, int blksize);
template<typename T>
int cuspread3d_nuptsdriven_prop(int nf1, int nf2, int nf3, int M,
                                cufinufft_plan_t<T> *d_plan);
template<typename T>
int cuspread3d_nuptsdriven(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
                           int blksize);
template<typename T>
int cuspread3d_blockgather_prop(int nf1, int nf2, int nf3, int M,
                                cufinufft_plan_t<T> *d_plan);
template<typename T>
int cuspread3d_blockgather(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
                           int blksize);
template<typename T>
int cuspread3d_subprob_prop(int nf1, int nf2, int nf3, int M,
                            cufinufft_plan_t<T> *d_plan);
template<typename T>
int cuspread3d_subprob(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
                       int blksize);

// Wrappers for methods of interpolation
template<typename T>
int cuinterp1d_nuptsdriven(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize);
template<typename T>
int cuinterp2d_nuptsdriven(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan,
                           int blksize);
template<typename T>
int cuinterp2d_subprob(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan, int blksize);
template<typename T>
int cuinterp3d_nuptsdriven(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
                           int blksize);
template<typename T>
int cuinterp3d_subprob(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
                       int blksize);

} // namespace spreadinterp
} // namespace cufinufft
#endif
