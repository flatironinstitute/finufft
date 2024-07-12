#include <cmath>
#include <iostream>

#include <cuda.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <thrust/extrema.h>

#include <cuda/std/complex>
#include <cufinufft/defs.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/utils.h>

#include <thrust/sort.h>

using namespace cufinufft::utils;

namespace cufinufft {
namespace spreadinterp {
/* ------------------------ 1d Spreading Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */

template<typename T, int KEREVALMETH>
__global__ void spread_1d_nuptsdriven(const T *x, const cuda_complex<T> *c,
                                      cuda_complex<T> *fw, int M, int ns, int nf1, T es_c,
                                      T es_beta, T sigma, const int *idxnupts) {

  auto ker1 = (T __restrict__ *)alloca(sizeof(T) * ns);

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    const auto x_rescaled     = fold_rescale(x[idxnupts[i]], nf1);
    const auto cnow           = c[idxnupts[i]];
    const auto [xstart, xend] = [ns, x_rescaled]() constexpr noexcept {
      if constexpr (std::is_same_v<T, float>) {
        const auto xstart = __float2int_ru(__fmaf_ru(ns, -.5f, x_rescaled));
        const auto xend   = __float2int_rd(__fmaf_rd(ns, .5f, x_rescaled));
        return int2{xstart, xend};
      }
      if constexpr (std::is_same_v<T, double>) {
        const auto xstart = __double2int_ru(__fma_ru(ns, -.5, x_rescaled));
        const auto xend   = __double2int_rd(__fma_rd(ns, .5, x_rescaled));
        return int2{xstart, xend};
      }
    }();
    const T x1 = (T)xstart - x_rescaled;
    if constexpr (KEREVALMETH == 1)
      eval_kernel_vec_horner(ker1, x1, ns, sigma);
    else
      eval_kernel_vec(ker1, x1, ns, es_c, es_beta);

    for (auto xx = xstart; xx <= xend; xx++) {
      auto ix    = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
      T kervalue = ker1[xx - xstart];
      atomicAdd(&fw[ix].x, cnow.x * kervalue);
      atomicAdd(&fw[ix].y, cnow.y * kervalue);
    }
  }
}

/* Kernels for SubProb Method */
// SubProb properties
template<typename T>
__global__ void calc_bin_size_noghost_1d(int M, int nf1, int bin_size_x, int nbinx,
                                         int *bin_size, const T *x, int *sortidx) {
  int binx;
  int oldidx;
  T x_rescaled;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    x_rescaled = fold_rescale(x[i], nf1);
    binx       = floor(x_rescaled / bin_size_x);
    binx       = binx >= nbinx ? binx - 1 : binx;
    binx       = binx < 0 ? 0 : binx;
    oldidx     = atomicAdd(&bin_size[binx], 1);
    sortidx[i] = oldidx;
    if (binx >= nbinx) {
      sortidx[i] = -binx;
    }
  }
}

template<typename T>
__global__ void calc_inverse_of_global_sort_idx_1d(
    int M, int bin_size_x, int nbinx, const int *bin_startpts, const int *sortidx,
    const T *x, int *index, int nf1) {
  int binx;
  T x_rescaled;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    x_rescaled = fold_rescale(x[i], nf1);
    binx       = floor(x_rescaled / bin_size_x);
    binx       = binx >= nbinx ? binx - 1 : binx;
    binx       = binx < 0 ? 0 : binx;

    index[bin_startpts[binx] + sortidx[i]] = i;
  }
}

template<typename T>
__forceinline__ __device__ cuda_complex<T> mul(const cuda_complex<T> &a, const T b) {
  return {a.x * b, a.y * b};
}

template<typename T, int KEREVALMETH>
__global__ void spread_1d_subprob(
    const T *x, const cuda_complex<T> *c, cuda_complex<T> *fw, int M, uint8_t ns, int nf1,
    T es_c, T es_beta, T sigma, const int *binstartpts, const int *bin_size,
    int bin_size_x, const int *subprob_to_bin, const int *subprobstartpts,
    const int *numsubprob, int maxsubprobsize, int nbinx, int *idxnupts) {
  extern __shared__ char sharedbuf[];
  alignas(256) auto *__restrict__ fwshared = (cuda_complex<T> *)sharedbuf;

  int ix;
  const int subpidx     = blockIdx.x;
  const int bidx        = subprob_to_bin[subpidx];
  const int binsubp_idx = subpidx - subprobstartpts[bidx];
  const int ptstart     = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
  const int nupts   = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);
  const int xoffset = (bidx % nbinx) * bin_size_x;
  const auto ns_2   = (ns + 1) / 2;
  const int N       = bin_size_x + 2 * ns_2;

  // dynamic stack allocation
  auto ker1 = (T __restrict__ *)alloca(sizeof(T) * ns);

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    fwshared[i] = {0, 0};
  }
  __syncthreads();

  for (auto i = threadIdx.x; i < nupts; i += blockDim.x) {
    const auto idx        = ptstart + i;
    const auto x_rescaled = fold_rescale(x[idxnupts[idx]], nf1);
    const auto cnow       = c[idxnupts[idx]];

    const auto [xstart, xend] = [ns, x_rescaled]() constexpr noexcept {
      if constexpr (std::is_same_v<T, float>) {
        const auto xstart = __float2int_ru(__fmaf_ru(ns, -.5f, x_rescaled));
        const auto xend   = __float2int_rd(__fmaf_rd(ns, .5f, x_rescaled));
        return int2{xstart, xend};
      }
      if constexpr (std::is_same_v<T, double>) {
        const auto xstart = __double2int_ru(__fma_ru(ns, -.5, x_rescaled));
        const auto xend   = __double2int_rd(__fma_rd(ns, .5, x_rescaled));
        return int2{xstart, xend};
      }
    }();

    const T x1 = T(xstart + xoffset) - x_rescaled;
    if constexpr (KEREVALMETH == 1)
      eval_kernel_vec_horner(ker1, x1, ns, sigma);
    else
      eval_kernel_vec(ker1, x1, ns, es_c, es_beta);
    for (int xx = xstart; xx <= xend; xx++) {
      ix = xx + ns_2;
      if (ix >= (bin_size_x + ns_2) || ix < 0) break;
      const auto result = mul(cnow, ker1[xx - xstart]);
      atomicAdd(&fwshared[ix].x, result.x);
      atomicAdd(&fwshared[ix].y, result.y);
    }
  }
  __syncthreads();
  /* write to global memory */
  for (int k = threadIdx.x; k < N; k += blockDim.x) {
    ix = xoffset - ns_2 + k;
    if (ix < (nf1 + ns_2)) {
      ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      atomicAdd(&fw[ix].x, fwshared[k].x);
      atomicAdd(&fw[ix].y, fwshared[k].y);
    }
  }
}

/* --------------------- 1d Interpolation Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */
template<typename T, int KEREVALMETH>
__global__ void interp_1d_nuptsdriven(const T *x, cuda_complex<T> *c,
                                      const cuda_complex<T> *fw, int M, int ns, int nf1,
                                      T es_c, T es_beta, T sigma, const int *idxnupts) {
  T ker1[MAX_NSPREAD];
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    T x_rescaled = fold_rescale(x[idxnupts[i]], nf1);

    int xstart = ceil(x_rescaled - ns / 2.0);
    int xend   = floor(x_rescaled + ns / 2.0);
    cuda_complex<T> cnow;
    cnow.x = 0.0;
    cnow.y = 0.0;

    T x1 = (T)xstart - x_rescaled;
    if constexpr (KEREVALMETH == 1)
      eval_kernel_vec_horner(ker1, x1, ns, sigma);
    else
      eval_kernel_vec(ker1, x1, ns, es_c, es_beta);

    for (int xx = xstart; xx <= xend; xx++) {
      int ix      = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
      T kervalue1 = ker1[xx - xstart];
      cnow.x += fw[ix].x * kervalue1;
      cnow.y += fw[ix].y * kervalue1;
    }
    c[idxnupts[i]].x = cnow.x;
    c[idxnupts[i]].y = cnow.y;
  }
}

} // namespace spreadinterp
} // namespace cufinufft
