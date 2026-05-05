#ifndef __CUSPREADINTERP_H__
#define __CUSPREADINTERP_H__

#include <cmath>
#include <cuda.h>
#include <thrust/sequence.h>

#include <cufinufft/common.hpp>
#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/contrib/helper_math.h>
#include <cufinufft/cufinufft_plan_t.hpp>
#include <cufinufft/intrinsics.hpp>
#include <cufinufft/utils.hpp>
#include <finufft_common/spread_opts.h>

namespace cufinufft {
namespace spreadinterp {

using namespace cufinufft::utils;
using namespace cufinufft::common;

// ES kernel reference evaluator used by host-side FT-quadrature code in
// src/cuda/common.cu. Kept here (rather than in the moved section below)
// because it takes a runtime spopts and is not on the device hot path.
template<typename T>
inline T evaluate_kernel(T x, const finufft_spread_opts &spopts)
/* ES ("exp sqrt" or "exp semicircle") kernel evaluation, single real argument:
   returns phi(2x/ns) := exp(beta.[sqrt(1 - (2x/ns)^2) - 1]),  for |x| < ns/2.
   This is the reference implementation, used by src/cuda/common.cu for onedim
   FT quadrature approx, so it need not be fast. */
{
  T z = 2.0 * x / T(spopts.nspread); // argument on [-1,1]
  if (abs(z) >= 1.0) return 0.0;
  return exp(T(spopts.beta) * (sqrt(T(1.0) - z * z) - T(1.0)));
}

/* --------------------------- Shared Helpers ---------------------------- */

template<typename T>
constexpr __forceinline__ __device__ T cudaFMA(const T a, const T b, const T c) {
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

template<typename T, int ns>
__forceinline__ __device__ T evaluate_kernel(T x, T es_c, T es_beta)
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
__inline__ __device__ void eval_kernel_vec(T *ker, const T x, const T es_c,
                                           const T es_beta) {
  // Eval the above direct ES kernel evaluator for arguments x+j, for j=0,..,w-1.
  // This is used when gpu_kerevalmeth=0.
  // Serves the same purpose as the below function eval_kernel_vec_horner.
  for (int i = 0; i < w; i++) ker[i] = evaluate_kernel<T, w>(abs(x + i), es_c, es_beta);
}

template<typename T, int w>
__device__ void eval_kernel_vec_horner(T *ker, const T x, const double upsampfac)
/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
   x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
   Two upsampfacs implemented (same as CPU coeffs created in 2018, used to 2025).
   The parameter (w and beta) choice in setup_spreadinterp must match these coeffs.
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

// Given grid sizes (via nf123) and bin sizes, compute the number of bins
// along every axis.
template<int ndim>
inline __host__ __device__ auto get_nbins(cuda::std::array<int, 3> nf123,
                                          cuda::std::array<int, 3> binsizes) {
  cuda::std::array<int, 3> nbins{1, 1, 1};
  for (int idim = 0; idim < ndim; ++idim)
    nbins[idim] = (nf123[idim] + binsizes[idim] - 1) / binsizes[idim];
  return nbins;
}

inline __host__ __device__ int nbins_total(const cuda::std::array<int, 3> &nbins) {
  return nbins[0] * nbins[1] * nbins[2];
}

// For the current nonuniform point (given via idx), compute the set of
// start indices of the spreading/interpolation area in the locally stored subgrid.
// Also compute the kernel values to use in spreading/interpolation.
template<typename T, int KEREVALMETH, int ndim, int ns>
__device__ auto get_kerval_and_local_start(
    int idx, cuda::std::array<const T *, 3> xyz, cuda::std::array<int, 3> nf,
    cuda::std::array<int, ndim> offset, T sigma, T es_c, T es_beta) {
  constexpr auto ns_2f = T(ns * .5);
  cuda::std::array<cuda::std::array<T, ns>, ndim> ker;
  cuda::std::array<int, ndim> start;
  for (int idim = 0; idim < ndim; ++idim) {
    const auto rescaled = fold_rescale(loadReadOnly(xyz[idim] + idx), nf[idim]);
    const auto s        = int(std::ceil(rescaled - ns_2f));
    if constexpr (KEREVALMETH == 1) {
      eval_kernel_vec_horner<T, ns>(&ker[idim][0], T(s) - rescaled, sigma);
    } else {
      eval_kernel_vec<T, ns>(&ker[idim][0], T(s) - rescaled, es_c, es_beta);
    }
    start[idim] = s - offset[idim];
  }
  return cuda::std::make_tuple(ker, start);
}

// For the current nonuniform point (given via idx), compute the set of
// start indices of the spreading/interpolation area in the locally stored subgrid.
// Also compute the kernel values to use in spreading/interpolation.
// (Version for nonunifom-points-driven algorithm.)
template<typename T, int KEREVALMETH, int ndim, int ns>
__device__ auto get_kerval_and_startpos_nuptsdriven(
    int idx, cuda::std::array<const T *, 3> xyz, cuda::std::array<int, 3> nf, T sigma,
    T es_c, T es_beta) {
  constexpr auto ns_2f = T(ns * .5);
  cuda::std::array<cuda::std::array<T, ns>, ndim> ker;
  cuda::std::array<int, ndim> start;
  for (size_t idim = 0; idim < ndim; ++idim) {
    const auto rescaled = fold_rescale(loadReadOnly(xyz[idim] + idx), nf[idim]);
    const auto s        = int(std::ceil(rescaled - ns_2f));
    if constexpr (KEREVALMETH == 1) {
      eval_kernel_vec_horner<T, ns>(&ker[idim][0], T(s) - rescaled, sigma);
    } else {
      eval_kernel_vec<T, ns>(&ker[idim][0], T(s) - rescaled, es_c, es_beta);
    }
    start[idim] = s + ((s < 0) ? nf[idim] : 0);
  }
  return cuda::std::make_tuple(ker, start);
}

template<int ndim>
__device__ auto compute_offset(const int bidx, const cuda::std::array<int, 3> &nbins,
                               const cuda::std::array<int, 3> &binsizes) {
  cuda::std::array<int, ndim> offset;
  int tmp = bidx;
  for (int idim = 0; idim + 1 < ndim; ++idim) {
    offset[idim] = (tmp % nbins[idim]) * binsizes[idim];
    tmp /= nbins[idim];
  }
  // last dimension can be done more cheaply
  offset[ndim - 1] = tmp * binsizes[ndim - 1];
  return offset;
}

// For the current nonuniform point (given via idx), compute the flat index
// of the bin it falls into.
template<int ndim, typename T>
__device__ int compute_bin_index(
    int idx, cuda::std::array<int, 3> nf, cuda::std::array<T, 3> inv_binsizes,
    cuda::std::array<int, 3> nbins, cuda::std::array<const T *, 3> xyz) {
  int binidx = 0;
  int stride = 1;
  for (int idim = 0; idim < ndim; ++idim) {
    const T rescaled = fold_rescale(loadReadOnly(xyz[idim] + idx), nf[idim]);
    int bin          = floor(rescaled * inv_binsizes[idim]);
    bin              = bin >= nbins[idim] ? bin - 1 : bin;
    bin              = bin < 0 ? 0 : bin;
    binidx += bin * stride;
    stride *= nbins[idim];
  }
  return binidx;
}

// Given bin sizes and kernel support, compute the size of a padded subgrid.
template<int ndim, int ns>
__device__ auto get_padded_subgrid_info(const cuda::std::array<int, 3> &binsizes) {
  constexpr auto rounded_ns = ((ns + 1) / 2) * 2;
  cuda::std::array<int, ndim> padded_size;
  int total = 1;
  for (int idim = 0; idim < ndim; ++idim) {
    padded_size[idim] = binsizes[idim] + rounded_ns;
    total *= padded_size[idim];
  }
  return cuda::std::make_tuple(padded_size, total);
}

// Given a flat index in a local padded subgrid, compute its location in the global grid.
template<int ndim, int ns>
__device__ int output_index_from_flat_local_index(
    int flatidx, const cuda::std::array<int, ndim> &padded_size,
    const cuda::std::array<int, ndim> &offset, const cuda::std::array<int, 3> &nf) {
  constexpr auto ns_2 = (ns + 1) / 2;

  int outidx    = 0;
  int outstride = 1;
  for (int idim = 0; idim < ndim; ++idim) {
    int idx0 = flatidx % padded_size[idim];
    int idx  = idx0 + offset[idim] - ns_2;
    idx      = idx < 0 ? idx + nf[idim] : (idx >= nf[idim] ? idx - nf[idim] : idx);
    outidx += idx * outstride;
    outstride *= nf[idim];
    flatidx /= padded_size[idim];
  }
  return outidx;
}

/* ------------------------- Interpolation section ------------------------------ */

// Iterates over all locations in a local subgrid, and for all pairs of
// corresponding local and global pixels, calls the provided function.
// Useful for copying between global and local grids.
template<typename T, int ndim, int ns, typename Func>
__device__ void shared_mem_copy_helper(cuda::std::array<int, 3> binsizes,
                                       cuda::std::array<int, ndim> offset,
                                       cuda::std::array<int, 3> nf, Func func) {
  constexpr auto ns_2 = (ns + 1) / 2;

  auto [padded_size, N] = get_padded_subgrid_info<ndim, ns>(binsizes);

  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    bool in_region = true;
    int flatidx    = n;
    int globidx    = 0;
    int globstride = 1;
    for (int idim = 0; idim < ndim; ++idim) {
      int idx0 = flatidx % padded_size[idim];
      int idx  = idx0 + offset[idim] - ns_2;
      if (idx >= nf[idim] + ns_2) in_region = false;
      idx = idx < 0 ? idx + nf[idim] : (idx >= nf[idim] ? idx - nf[idim] : idx);
      globidx += idx * globstride;
      globstride *= nf[idim];
      flatidx /= padded_size[idim];
    }
    if (in_region)
      func(n, globidx); // atomicAddComplexGlobal<T>(fw + outidx, fwshared[n]);
  }
}

/* ------------------------- Spreading section ------------------------------ */

// FIXME unify the next two functions and templatize on a lambda?
template<typename T, int ndim>
__global__ FINUFFT_FLATTEN void calc_bin_size_noghost(
    const int M, const cuda::std::array<int, 3> nf,
    const cuda::std::array<T, 3> inv_binsizes, const cuda::std::array<int, 3> nbins,
    int *FINUFFT_RESTRICT bin_size, const cuda::std::array<const T *, 3> xyz,
    int *FINUFFT_RESTRICT sortidx) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    const int binidx = compute_bin_index<ndim>(i, nf, inv_binsizes, nbins, xyz);
    const int oldidx = atomicAdd(&bin_size[binidx], 1);
    storeCacheStreaming(sortidx + i, oldidx);
  }
}

template<typename T, int ndim>
__global__ FINUFFT_FLATTEN void calc_inverse_of_global_sort_idx(
    const int M, const cuda::std::array<T, 3> inv_binsizes,
    const cuda::std::array<int, 3> nbins, const int *FINUFFT_RESTRICT bin_startpts,
    const int *FINUFFT_RESTRICT sortidx, const cuda::std::array<const T *, 3> xyz,
    int *FINUFFT_RESTRICT index, const cuda::std::array<int, 3> nf) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    const int binidx = compute_bin_index<ndim>(i, nf, inv_binsizes, nbins, xyz);
    storeCacheStreaming(
        index + loadReadOnly(bin_startpts + binidx) + loadReadOnly(sortidx + i), i);
  }
}

// `static` (not `inline`) so each TU gets a uniquely-named kernel symbol;
// otherwise, with RDC off, every TU that includes this header registers a
// kernel with the same mangled name and the CUDA Runtime emits a
// "Duplicate entry kernels" warning at first launch (compute-sanitizer
// reports it as an error). Templated kernels above don't have the issue
// because vague-linkage lets the linker fold the duplicates.
static __global__ void calc_subprob(const int *FINUFFT_RESTRICT bin_size,
                                    int *FINUFFT_RESTRICT num_subprob,
                                    const int maxsubprobsize, const int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    num_subprob[i] = (loadReadOnly(bin_size + i) + maxsubprobsize - 1) / maxsubprobsize;
  }
}
static __global__ void map_b_into_subprob(
    int *FINUFFT_RESTRICT d_subprob_to_bin, const int *FINUFFT_RESTRICT d_subprobstartpts,
    const int *FINUFFT_RESTRICT d_numsubprob, const int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    for (int j = 0; j < loadReadOnly(d_numsubprob + i); j++) {
      d_subprob_to_bin[loadReadOnly(d_subprobstartpts + i) + j] = i;
    }
  }
}

/* ---- Per-method __global__ kernels and host drivers moved out ----
 * spread/interp kernels now live alongside their dispatch glue in
 * src/cuda/{spread,interp}_*.cuh.
 */

} // namespace spreadinterp
} // namespace cufinufft
#endif
