#pragma once

#include <cmath>

#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/contrib/helper_math.h>

#include <cuda/std/mdspan>
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

/* ------------------------ 2d Spreading Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */

template<typename T, int KEREVALMETH, int ns>
__global__ void spread_2d_nupts_driven(const T *x, const T *y, const cuda_complex<T> *c,
                                       cuda_complex<T> *fw, int M, int nf1, int nf2,
                                       T es_c, T es_beta, T sigma, const int *idxnupts) {
  T ker1[ns];
  T ker2[ns];
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    const auto x_rescaled     = fold_rescale(x[idxnupts[i]], nf1);
    const auto y_rescaled     = fold_rescale(y[idxnupts[i]], nf2);
    const auto cnow           = c[idxnupts[i]];
    const auto [xstart, xend] = interval(ns, x_rescaled);
    const auto [ystart, yend] = interval(ns, y_rescaled);

    const auto x1 = (T)xstart - x_rescaled;
    const auto y1 = (T)ystart - y_rescaled;

    if constexpr (KEREVALMETH == 1) {
      eval_kernel_vec_horner<T, ns>(ker1, x1, sigma);
      eval_kernel_vec_horner<T, ns>(ker2, y1, sigma);
    } else {
      eval_kernel_vec<T, ns>(ker1, x1, es_c, es_beta);
      eval_kernel_vec<T, ns>(ker2, y1, es_c, es_beta);
    }

    for (auto yy = ystart; yy <= yend; yy++) {
      const auto iy = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
      for (auto xx = xstart; xx <= xend; xx++) {
        const auto ix        = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
        const auto outidx    = ix + iy * nf1;
        const auto kervalue1 = ker1[xx - xstart];
        const auto kervalue2 = ker2[yy - ystart];
        const cuda_complex<T> res{cnow.x * kervalue1 * kervalue2,
                                  cnow.y * kervalue1 * kervalue2};
        atomicAddComplexGlobal<T>(fw + outidx, res);
      }
    }
  }
}

/* Kernels for SubProb Method */
// SubProb properties
template<typename T>
__global__ void calc_bin_size_noghost_2d(int M, int nf1, int nf2, int bin_size_x,
                                         int bin_size_y, int nbinx, int nbiny,
                                         int *bin_size, T *x, T *y, int *sortidx) {
  int binidx, binx, biny;
  int oldidx;
  T x_rescaled, y_rescaled;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    x_rescaled = fold_rescale(x[i], nf1);
    y_rescaled = fold_rescale(y[i], nf2);
    binx       = floor(x_rescaled / bin_size_x);
    binx       = binx >= nbinx ? binx - 1 : binx;
    binx       = binx < 0 ? 0 : binx;
    biny       = floor(y_rescaled / bin_size_y);
    biny       = biny >= nbiny ? biny - 1 : biny;
    biny       = biny < 0 ? 0 : biny;
    binidx     = binx + biny * nbinx;
    oldidx     = atomicAdd(&bin_size[binidx], 1);
    sortidx[i] = oldidx;
    if (binx >= nbinx || biny >= nbiny) {
      sortidx[i] = -biny;
    }
  }
}

template<typename T>
__global__ void calc_inverse_of_global_sort_index_2d(
    int M, int bin_size_x, int bin_size_y, int nbinx, int nbiny, const int *bin_startpts,
    const int *sortidx, const T *x, const T *y, int *index, int nf1, int nf2) {
  int binx, biny;
  int binidx;
  T x_rescaled, y_rescaled;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    x_rescaled = fold_rescale(x[i], nf1);
    y_rescaled = fold_rescale(y[i], nf2);
    binx       = floor(x_rescaled / bin_size_x);
    binx       = binx >= nbinx ? binx - 1 : binx;
    binx       = binx < 0 ? 0 : binx;
    biny       = floor(y_rescaled / bin_size_y);
    biny       = biny >= nbiny ? biny - 1 : biny;
    biny       = biny < 0 ? 0 : biny;
    binidx     = binx + biny * nbinx;

    index[bin_startpts[binidx] + sortidx[i]] = i;
  }
}

template<typename T, int KEREVALMETH, int ns>
__global__ void spread_2d_subprob(
    const T *x, const T *y, const cuda_complex<T> *c, cuda_complex<T> *fw, int M, int nf1,
    int nf2, T es_c, T es_beta, T sigma, int *binstartpts, const int *bin_size,
    int bin_size_x, int bin_size_y, int *subprob_to_bin, const int *subprobstartpts,
    const int *numsubprob, int maxsubprobsize, int nbinx, int nbiny,
    const int *idxnupts) {
  extern __shared__ char sharedbuf[];
  cuda_complex<T> *fwshared = (cuda_complex<T> *)sharedbuf;

  const int subpidx      = blockIdx.x;
  const auto bidx        = subprob_to_bin[subpidx];
  const auto binsubp_idx = subpidx - subprobstartpts[bidx];
  const auto ptstart     = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
  const auto nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

  const int xoffset = (bidx % nbinx) * bin_size_x;
  const int yoffset = (bidx / nbinx) * bin_size_y;

  const T ns_2f         = ns * T(.5);
  const auto ns_2       = (ns + 1) / 2;
  const auto rounded_ns = ns_2 * 2;
  const int N           = (bin_size_x + rounded_ns) * (bin_size_y + rounded_ns);

  T ker1[ns];
  T ker2[ns];

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    fwshared[i] = {0, 0};
  }
  __syncthreads();

  for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
    const int idx         = ptstart + i;
    const auto x_rescaled = fold_rescale(x[idxnupts[idx]], nf1);
    const auto y_rescaled = fold_rescale(y[idxnupts[idx]], nf2);
    const auto cnow       = c[idxnupts[idx]];
    auto [xstart, xend]   = interval(ns, x_rescaled);
    auto [ystart, yend]   = interval(ns, y_rescaled);

    const T x1 = T(xstart) - x_rescaled;
    const T y1 = T(ystart) - y_rescaled;
    xstart -= xoffset;
    ystart -= yoffset;
    xend -= xoffset;
    yend -= yoffset;

    if constexpr (KEREVALMETH == 1) {
      eval_kernel_vec_horner<T, ns>(ker1, x1, sigma);
      eval_kernel_vec_horner<T, ns>(ker2, y1, sigma);
    } else {
      eval_kernel_vec<T, ns>(ker1, x1, es_c, es_beta);
      eval_kernel_vec<T, ns>(ker2, y1, es_c, es_beta);
    }

    for (int yy = ystart; yy <= yend; yy++) {
      const auto iy = yy + ns_2;
      if (iy >= (bin_size_y + rounded_ns) || iy < 0) break;
      for (int xx = xstart; xx <= xend; xx++) {
        const auto ix = xx + ns_2;
        if (ix >= (bin_size_x + rounded_ns) || ix < 0) break;
        const auto outidx   = ix + iy * (bin_size_x + rounded_ns);
        const auto kervalue = ker1[xx - xstart] * ker2[yy - ystart];
        const cuda_complex<T> res{cnow.x * kervalue, cnow.y * kervalue};
        atomicAddComplexShared<T>(fwshared + outidx, res);
      }
    }
  }

  __syncthreads();
  /* write to global memory */
  for (int k = threadIdx.x; k < N; k += blockDim.x) {
    const auto i = k % (bin_size_x + rounded_ns);
    const auto j = k / (bin_size_x + rounded_ns);
    auto ix      = xoffset - ns_2 + i;
    auto iy      = yoffset - ns_2 + j;
    if (ix < (nf1 + ns_2) && iy < (nf2 + ns_2)) {
      ix                   = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      iy                   = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
      const auto outidx    = ix + iy * nf1;
      const auto sharedidx = i + j * (bin_size_x + rounded_ns);
      atomicAddComplexGlobal<T>(fw + outidx, fwshared[sharedidx]);
    }
  }
}

template<typename T, int KEREVALMETH, int ns>
__global__ void spread_2d_output_driven(
    const T *x, const T *y, const cuda_complex<T> *c, cuda_complex<T> *fw, int M, int nf1,
    int nf2, T es_c, T es_beta, T sigma, int *binstartpts, const int *bin_size,
    int bin_size_x, int bin_size_y, int *subprob_to_bin, const int *subprobstartpts,
    const int *numsubprob, int maxsubprobsize, int nbinx, int nbiny, const int *idxnupts,
    const int np) {
  extern __shared__ char sharedbuf[];

  static constexpr auto ns_2f      = T(ns * .5);
  static constexpr auto ns_2       = (ns + 1) / 2;
  static constexpr auto rounded_ns = ns_2 * 2;

  const auto padded_size_x = bin_size_x + rounded_ns;
  const auto padded_size_y = bin_size_y + rounded_ns;

  const int bidx        = subprob_to_bin[blockIdx.x];
  const int binsubp_idx = blockIdx.x - subprobstartpts[bidx];
  const int ptstart     = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
  const int nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

  const int xoffset = (bidx % nbinx) * bin_size_x;
  const int yoffset = ((bidx / nbinx) % nbiny) * bin_size_y;

  using mdspan_t = mdspan<T, extents<int, dynamic_extent, 2, ns>>;
  auto kerevals = mdspan_t((T *)sharedbuf, np);
  // sharedbuf + size of kerevals in bytes
  // Offset pointer into sharedbuf after kerevals
  // Create span using pointer + size

  auto nupts_sm = span(
      reinterpret_cast<cuda_complex<T> *>(kerevals.data_handle() + kerevals.size()),
      np);

  auto shift = span(reinterpret_cast<int2 *>(nupts_sm.data() + nupts_sm.size()), np);

  auto local_subgrid = mdspan<cuda_complex<T>, dextents<int, 2>>(
      reinterpret_cast<cuda_complex<T> *>(shift.data() + shift.size()), padded_size_y,
      padded_size_x);

  // set local_subgrid to zero
  for (int i = threadIdx.x; i < local_subgrid.size(); i += blockDim.x) {
    local_subgrid.data_handle()[i] = {0, 0};
  }
  __syncthreads();

  for (int batch_begin = 0; batch_begin < nupts; batch_begin += np) {
    const auto batch_size = min(np, nupts - batch_begin);
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
      const int nuptsidx = idxnupts[ptstart + i + batch_begin];
      // index of the current point within the batch
      const auto x_rescaled = fold_rescale(x[nuptsidx], nf1);
      const auto y_rescaled = fold_rescale(y[nuptsidx], nf2);
      nupts_sm[i]           = c[nuptsidx];
      auto [xstart, xend]   = interval(ns, x_rescaled);
      auto [ystart, yend]   = interval(ns, y_rescaled);
      const T x1            = T(xstart) - x_rescaled;
      const T y1            = T(ystart) - y_rescaled;

      shift[i] = {
          xstart - xoffset,
          ystart - yoffset,
      };

      if constexpr (KEREVALMETH == 1) {
        eval_kernel_vec_horner<T, ns>(&kerevals(i, 0, 0), x1, sigma);
        eval_kernel_vec_horner<T, ns>(&kerevals(i, 1, 0), y1, sigma);
      } else {
        eval_kernel_vec<T, ns>(&kerevals(i, 0, 0), x1, es_c, es_beta);
        eval_kernel_vec<T, ns>(&kerevals(i, 1, 0), y1, es_c, es_beta);
      }
    }
    __syncthreads();

    for (auto i = 0; i < batch_size; i++) {
      // strength from shared memory
      static constexpr int sizex  = ns; // true span in X
      const auto cnow             = nupts_sm[i];
      const auto [xstart, ystart] = shift[i];
      static constexpr auto total = ns * ns;

      for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        // decompose idx using `plane`
        const int yy = idx / sizex;
        const int xx = idx - yy * sizex;

        // recover global coords
        const int real_yy = ystart + yy;
        const int real_xx = xstart + xx;

        // padded indices
        const int iy = real_yy + ns_2;
        const int ix = real_xx + ns_2;

        // separable window weights
        const auto kervalue = kerevals(i, 0, xx) * kerevals(i, 1, yy);

        // accumulate
        local_subgrid(iy, ix) += {cnow * kervalue};
      }
      __syncthreads();
    }
  }
  for (int n = threadIdx.x; n < local_subgrid.size(); n += blockDim.x) {
    const int i = n % (padded_size_x);
    const int j = n / (padded_size_x);

    int ix = xoffset - ns_2 + i;
    int iy = yoffset - ns_2 + j;

    if (ix < (nf1 + ns_2) && iy < (nf2 + ns_2)) {
      ix               = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      iy               = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
      const int outidx = ix + iy * nf1;
      atomicAddComplexGlobal<T>(fw + outidx, local_subgrid(j, i));
    }
  }
}

/* --------------------- 2d Interpolation Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */
template<typename T, int KEREVALMETH, int ns>
__global__ void interp_2d_nupts_driven(const T *x, const T *y, cuda_complex<T> *c,
                                       const cuda_complex<T> *fw, int M, int nf1, int nf2,
                                       T es_c, T es_beta, T sigma, const int *idxnupts) {
  T ker1[ns];
  T ker2[ns];

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    const auto x_rescaled     = fold_rescale(x[idxnupts[i]], nf1);
    const auto y_rescaled     = fold_rescale(y[idxnupts[i]], nf2);
    const auto [xstart, xend] = interval(ns, x_rescaled);
    const auto [ystart, yend] = interval(ns, y_rescaled);

    T x1 = (T)xstart - x_rescaled;
    T y1 = (T)ystart - y_rescaled;

    if constexpr (KEREVALMETH == 1) {
      eval_kernel_vec_horner<T, ns>(ker1, x1, sigma);
      eval_kernel_vec_horner<T, ns>(ker2, y1, sigma);
    } else {
      eval_kernel_vec<T, ns>(ker1, x1, es_c, es_beta);
      eval_kernel_vec<T, ns>(ker2, y1, es_c, es_beta);
    }

    cuda_complex<T> cnow{0, 0};
    for (int yy = ystart; yy <= yend; yy++) {
      const T kervalue2 = ker2[yy - ystart];
      const auto iy     = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
      for (int xx = xstart; xx <= xend; xx++) {
        const auto ix        = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
        const auto inidx     = ix + iy * nf1;
        const auto kervalue1 = ker1[xx - xstart];
        cnow.x += fw[inidx].x * kervalue1 * kervalue2;
        cnow.y += fw[inidx].y * kervalue1 * kervalue2;
      }
    }
    c[idxnupts[i]] = cnow;
  }
}

/* Kernels for Subprob Method */
template<typename T, int KEREVALMETH, int ns>
__global__ void interp_2d_subprob(
    const T *x, const T *y, cuda_complex<T> *c, const cuda_complex<T> *fw, int M, int nf1,
    int nf2, T es_c, T es_beta, T sigma, int *binstartpts, const int *bin_size,
    int bin_size_x, int bin_size_y, int *subprob_to_bin, const int *subprobstartpts,
    const int *numsubprob, int maxsubprobsize, int nbinx, int nbiny,
    const int *idxnupts) {
  extern __shared__ char sharedbuf[];
  cuda_complex<T> *fwshared = (cuda_complex<T> *)sharedbuf;

  T ker1[ns];
  T ker2[ns];

  const auto subpidx     = blockIdx.x;
  const auto bidx        = subprob_to_bin[subpidx];
  const auto binsubp_idx = subpidx - subprobstartpts[bidx];
  const auto ptstart     = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
  const auto nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

  const auto xoffset = (bidx % nbinx) * bin_size_x;
  const auto yoffset = (bidx / nbinx) * bin_size_y;

  const T ns_2f         = ns * T(.5);
  const auto ns_2       = (ns + 1) / 2;
  const auto rounded_ns = ns_2 * 2;
  const int N           = (bin_size_x + rounded_ns) * (bin_size_y + rounded_ns);

  for (int k = threadIdx.x; k < N; k += blockDim.x) {
    int i   = k % (bin_size_x + rounded_ns);
    int j   = k / (bin_size_x + rounded_ns);
    auto ix = xoffset - ns_2 + i;
    auto iy = yoffset - ns_2 + j;
    if (ix < (nf1 + ns_2) && iy < (nf2 + ns_2)) {
      ix                   = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      iy                   = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
      const auto outidx    = ix + iy * nf1;
      const auto sharedidx = i + j * (bin_size_x + rounded_ns);
      fwshared[sharedidx]  = fw[outidx];
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
    int idx               = ptstart + i;
    const auto x_rescaled = fold_rescale(x[idxnupts[idx]], nf1);
    const auto y_rescaled = fold_rescale(y[idxnupts[idx]], nf2);
    cuda_complex<T> cnow{0, 0};

    auto [xstart, xend] = interval(ns, x_rescaled);
    auto [ystart, yend] = interval(ns, y_rescaled);

    const T x1 = T(xstart) - x_rescaled;
    const T y1 = T(ystart) - y_rescaled;

    xstart -= xoffset;
    ystart -= yoffset;
    xend -= xoffset;
    yend -= yoffset;

    if constexpr (KEREVALMETH == 1) {
      eval_kernel_vec_horner<T, ns>(ker1, x1, sigma);
      eval_kernel_vec_horner<T, ns>(ker2, y1, sigma);
    } else {
      eval_kernel_vec<T, ns>(ker1, x1, es_c, es_beta);
      eval_kernel_vec<T, ns>(ker2, y1, es_c, es_beta);
    }

    for (int yy = ystart; yy <= yend; yy++) {
      const auto kervalue2 = ker2[yy - ystart];
      for (int xx = xstart; xx <= xend; xx++) {
        const auto ix        = xx + ns_2;
        const auto iy        = yy + ns_2;
        const auto outidx    = ix + iy * (bin_size_x + rounded_ns);
        const auto kervalue1 = ker1[xx - xstart];
        cnow.x += fwshared[outidx].x * kervalue1 * kervalue2;
        cnow.y += fwshared[outidx].y * kervalue1 * kervalue2;
      }
    }
    c[idxnupts[idx]] = cnow;
  }
}

} // namespace spreadinterp
} // namespace cufinufft
