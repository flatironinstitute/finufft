#pragma once

#include <cuda/std/mdspan>

#include <cmath>
#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/contrib/helper_math.h>

#include <cufinufft/spreadinterp.h>
#include <cufinufft/utils.h>

using namespace cufinufft::utils;

namespace cufinufft {
namespace spreadinterp {

using cuda::std::dextents;
using cuda::std::dynamic_extent;
using cuda::std::extents;
using cuda::std::mdspan;
using cuda::std::span;
/* ------------------------ 1d Spreading Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */

template<typename T, int KEREVALMETH, int ns>
__global__ void spread_1d_nuptsdriven(const T *x, const cuda_complex<T> *c,
                                      cuda_complex<T> *fw, int M, int nf1, T es_c,
                                      T es_beta, T sigma, const int *idxnupts) {
  T ker1[ns];
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    const auto x_rescaled     = fold_rescale(x[idxnupts[i]], nf1);
    const auto cnow           = c[idxnupts[i]];
    const auto [xstart, xend] = interval(ns, x_rescaled);
    const T x1                = (T)xstart - x_rescaled;
    if constexpr (KEREVALMETH == 1)
      eval_kernel_vec_horner<T, ns>(ker1, x1, sigma);
    else
      eval_kernel_vec<T, ns>(ker1, x1, es_c, es_beta);

    for (auto xx = xstart; xx <= xend; xx++) {
      auto ix          = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
      const T kervalue = ker1[xx - xstart];
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

template<typename T, int KEREVALMETH, int ns>
__global__ void spread_1d_output_driven(
    const T *x, const cuda_complex<T> *c, cuda_complex<T> *fw, int M, int nf1, T es_c,
    T es_beta, T sigma, const int *binstartpts, const int *bin_size, int bin_size_x,
    const int *subprob_to_bin, const int *subprobstartpts, const int *numsubprob,
    int maxsubprobsize, int nbinx, int *idxnupts, const int np) {
  extern __shared__ char sharedbuf[];

  static constexpr auto ns_2f      = T(ns * .5);
  static constexpr auto ns_2       = (ns + 1) / 2;
  static constexpr auto rounded_ns = ns_2 * 2;

  const auto padded_size_x = bin_size_x + rounded_ns;

  const int bidx        = subprob_to_bin[blockIdx.x];
  const int binsubp_idx = blockIdx.x - subprobstartpts[bidx];
  const int ptstart     = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
  const int nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

  const int xoffset = (bidx % nbinx) * bin_size_x;

  using mdspan_t   = mdspan<T, extents<int, dynamic_extent, ns>>;
  auto window_vals = mdspan_t((T *)sharedbuf, np);
  // sharedbuf + size of window_vals in bytes
  // Offset pointer into sharedbuf after window_vals
  // Create span using pointer + size

  auto vp_sm = span(
      reinterpret_cast<cuda_complex<T> *>(window_vals.data_handle() + window_vals.size()),
      np);

  auto shift = span(reinterpret_cast<int *>(vp_sm.data() + vp_sm.size()), np);

  auto u_local = span<cuda_complex<T>>(
      reinterpret_cast<cuda_complex<T> *>(shift.data() + shift.size()), padded_size_x);

  // set u_local to zero
  for (int i = threadIdx.x; i < u_local.size(); i += blockDim.x) {
    u_local[i] = {0, 0};
  }
  __syncthreads();

  for (int batch_begin = 0; batch_begin < nupts; batch_begin += np) {
    const auto batch_size = min(np, nupts - batch_begin);
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
      const int nuptsidx = idxnupts[ptstart + i + batch_begin];
      // index of the current point within the batch
      const auto x_rescaled = fold_rescale(x[nuptsidx], nf1);
      vp_sm[i]              = c[nuptsidx];
      auto [xstart, xend]   = interval(ns, x_rescaled);
      const T x1            = T(xstart) - x_rescaled;

      shift[i] = xstart - xoffset;

      if constexpr (KEREVALMETH == 1) {
        eval_kernel_vec_horner<T, ns>(&window_vals(i, 0), x1, sigma);
      } else {
        eval_kernel_vec<T, ns>(&window_vals(i, 0), x1, es_c, es_beta);
      }
    }
    __syncthreads();

    for (auto i = 0; i < batch_size; i++) {
      // strength from shared memory

      const auto cnow             = vp_sm[i];
      const auto xstart           = shift[i];
      static constexpr auto total = ns;
      for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int ix = xstart + idx + ns_2;
        // separable window weights
        const auto kervalue = window_vals(i, idx);

        // accumulate
        const cuda_complex<T> res{cnow.x * kervalue, cnow.y * kervalue};
        u_local[ix] += res;
      }
      __syncthreads();
    }
  }
  /* write to global memory */
  for (int k = threadIdx.x; k < u_local.size(); k += blockDim.x) {
    auto ix = xoffset - ns_2 + k;
    if (ix < (nf1 + ns_2)) {
      ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      atomicAddComplexGlobal<T>(fw + ix, u_local[k]);
    }
  }
}

template<typename T, int KEREVALMETH, int ns>
__global__ void spread_1d_subprob(
    const T *x, const cuda_complex<T> *c, cuda_complex<T> *fw, int M, int nf1, T es_c,
    T es_beta, T sigma, const int *binstartpts, const int *bin_size, int bin_size_x,
    const int *subprob_to_bin, const int *subprobstartpts, const int *numsubprob,
    int maxsubprobsize, int nbinx, int *idxnupts) {
  extern __shared__ char sharedbuf[];
  auto *__restrict__ fwshared = (cuda_complex<T> *)sharedbuf;

  const int subpidx     = blockIdx.x;
  const int bidx        = subprob_to_bin[subpidx];
  const int binsubp_idx = subpidx - subprobstartpts[bidx];
  const int ptstart     = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
  const int nupts   = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);
  const int xoffset = (bidx % nbinx) * bin_size_x;
  const auto ns_2   = (ns + 1) / 2;
  const auto rounded_ns = ns_2 * 2;
  const int N           = bin_size_x + rounded_ns;

  T ker1[ns];

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    fwshared[i] = {0, 0};
  }

  const T ns_2f = ns * T(.5);

  __syncthreads();

  for (auto i = threadIdx.x; i < nupts; i += blockDim.x) {
    const auto idx        = ptstart + i;
    const auto x_rescaled = fold_rescale(x[idxnupts[idx]], nf1);
    const auto cnow       = c[idxnupts[idx]];
    auto [xstart, xend]   = interval(ns, x_rescaled);

    const T x1 = T(xstart) - x_rescaled;
    xstart -= xoffset;
    xend -= xoffset;

    if constexpr (KEREVALMETH == 1)
      eval_kernel_vec_horner<T, ns>(ker1, x1, sigma);
    else
      eval_kernel_vec<T, ns>(ker1, x1, es_c, es_beta);
    for (int xx = xstart; xx <= xend; xx++) {
      const auto ix = xx + ns_2;
      if (ix >= (bin_size_x + rounded_ns) || ix < 0) break;
      const cuda_complex<T> result{cnow.x * ker1[xx - xstart],
                                   cnow.y * ker1[xx - xstart]};
      atomicAddComplexShared<T>(fwshared + ix, result);
    }
  }
  __syncthreads();
  /* write to global memory */
  for (int k = threadIdx.x; k < N; k += blockDim.x) {
    auto ix = xoffset - ns_2 + k;
    if (ix < (nf1 + ns_2)) {
      ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      atomicAddComplexGlobal<T>(fw + ix, fwshared[k]);
    }
  }
}

/* --------------------- 1d Interpolation Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */
template<typename T, int KEREVALMETH, int ns>
__global__ void interp_1d_nuptsdriven(const T *x, cuda_complex<T> *c,
                                      const cuda_complex<T> *fw, int M, int nf1, T es_c,
                                      T es_beta, T sigma, const int *idxnupts) {

  T ker1[ns];

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    const T x_rescaled        = fold_rescale(x[idxnupts[i]], nf1);
    const auto [xstart, xend] = interval(ns, x_rescaled);

    cuda_complex<T> cnow{0, 0};

    const T x1 = (T)xstart - x_rescaled;
    if constexpr (KEREVALMETH == 1)
      eval_kernel_vec_horner<T, ns>(ker1, x1, sigma);
    else
      eval_kernel_vec<T, ns>(ker1, x1, es_c, es_beta);
    for (int xx = xstart; xx <= xend; xx++) {
      int ix            = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
      const T kervalue1 = ker1[xx - xstart];
      cnow.x += fw[ix].x * kervalue1;
      cnow.y += fw[ix].y * kervalue1;
    }
    c[idxnupts[i]] = cnow;
  }
}

} // namespace spreadinterp
} // namespace cufinufft
