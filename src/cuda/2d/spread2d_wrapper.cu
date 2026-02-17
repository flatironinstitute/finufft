#include <cassert>
#include <iostream>

#include <cuComplex.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <cufinufft/contrib/helper_math.h>
#include <cufinufft/common.h>
#include <cufinufft/precision_independent.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/utils.h>

using namespace cufinufft::common;
using namespace cufinufft::utils;

namespace cufinufft {
namespace spreadinterp {

using cuda::std::dextents;
using cuda::std::dynamic_extent;
using cuda::std::extents;
using cuda::std::mdspan;
using cuda::std::span;

template<typename T>
static __global__ void calc_bin_size_noghost_2d(int M, int nf1, int nf2, int bin_size_x,
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
static __global__ void calc_inverse_of_global_sort_index_2d(
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
static __global__ void spread_2d_nupts_driven(const T *x, const T *y, const cuda_complex<T> *c,
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

template<typename T, int KEREVALMETH, int ns>
static __global__ void spread_2d_output_driven(
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

        if constexpr (std::is_same_v<T, float>) {
          if (ix >= (padded_size_x) || ix < 0) break;
          if (iy >= (padded_size_y) || iy < 0) break;
        }
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

template<typename T, int KEREVALMETH, int ns>
static __global__ void spread_2d_subprob(
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

template<typename T, int ns>
static int cuspread2d_output_driven(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan,
                             int blksize) {
  auto &stream = d_plan->stream;

  T es_c             = 4.0 / T(d_plan->spopts.nspread * d_plan->spopts.nspread);
  T es_beta          = d_plan->spopts.beta;
  int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;

  // assume that bin_size_x > ns/2;
  int bin_size_x = d_plan->opts.gpu_binsizex;
  int bin_size_y = d_plan->opts.gpu_binsizey;
  int numbins[2];
  numbins[0] = ceil((T)nf1 / bin_size_x);
  numbins[1] = ceil((T)nf2 / bin_size_y);

  T *d_kx               = d_plan->kxyz[0];
  T *d_ky               = d_plan->kxyz[1];
  cuda_complex<T> *d_c  = d_plan->c;
  cuda_complex<T> *d_fw = d_plan->fw;

  int *d_binsize         = d_plan->binsize;
  int *d_binstartpts     = d_plan->binstartpts;
  int *d_numsubprob      = d_plan->numsubprob;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_idxnupts        = d_plan->idxnupts;

  int totalnumsubprob   = d_plan->totalnumsubprob;
  int *d_subprob_to_bin = d_plan->subprob_to_bin;

  T sigma = d_plan->opts.upsampfac;

  const auto sharedplanorysize = shared_memory_required<T>(
      2, d_plan->spopts.nspread, d_plan->opts.gpu_binsizex, d_plan->opts.gpu_binsizey,
      d_plan->opts.gpu_binsizez, d_plan->opts.gpu_np);

  if (d_plan->opts.gpu_kerevalmeth) {
    if (const auto finufft_err =
            cufinufft_set_shared_memory(spread_2d_output_driven<T, 1, ns>, 2, *d_plan) !=
            0) {
      return FINUFFT_ERR_INSUFFICIENT_SHMEM;
    }
    for (int t = 0; t < blksize; t++) {
      spread_2d_output_driven<T, 1, ns>
          <<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
              d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, nf1, nf2, es_c, es_beta,
              sigma, d_binstartpts, d_binsize, bin_size_x, bin_size_y, d_subprob_to_bin,
              d_subprobstartpts, d_numsubprob, maxsubprobsize, numbins[0], numbins[1],
              d_idxnupts, d_plan->opts.gpu_np);
      RETURN_IF_CUDA_ERROR
    }
  } else {
    if (const auto finufft_err =
            cufinufft_set_shared_memory(spread_2d_subprob<T, 0, ns>, 2, *d_plan) != 0) {
      return FINUFFT_ERR_INSUFFICIENT_SHMEM;
    }
    for (int t = 0; t < blksize; t++) {
      spread_2d_output_driven<T, 0, ns>
          <<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
              d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, nf1, nf2, es_c, es_beta,
              sigma, d_binstartpts, d_binsize, bin_size_x, bin_size_y, d_subprob_to_bin,
              d_subprobstartpts, d_numsubprob, maxsubprobsize, numbins[0], numbins[1],
              d_idxnupts, d_plan->opts.gpu_np);
      RETURN_IF_CUDA_ERROR
    }
  }
  return 0;
}

template<typename T, int ns>
static int cuspread2d_nuptsdriven(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan,
                           int blksize) {
  auto &stream = d_plan->stream;
  dim3 threadsPerBlock;
  dim3 blocks;

  int *d_idxnupts = d_plan->idxnupts;
  T es_c          = 4.0 / T(d_plan->spopts.nspread * d_plan->spopts.nspread);
  T es_beta       = d_plan->spopts.beta;
  T sigma         = d_plan->spopts.upsampfac;

  T *d_kx               = d_plan->kxyz[0];
  T *d_ky               = d_plan->kxyz[1];
  cuda_complex<T> *d_c  = d_plan->c;
  cuda_complex<T> *d_fw = d_plan->fw;

  threadsPerBlock.x = 16;
  threadsPerBlock.y = 1;
  blocks.x          = (M + threadsPerBlock.x - 1) / threadsPerBlock.x;
  blocks.y          = 1;
  if (d_plan->opts.gpu_kerevalmeth) {
    for (int t = 0; t < blksize; t++) {
      spread_2d_nupts_driven<T, 1, ns><<<blocks, threadsPerBlock, 0, stream>>>(
          d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, nf1, nf2, es_c, es_beta,
          sigma, d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  } else {
    for (int t = 0; t < blksize; t++) {
      spread_2d_nupts_driven<T, 0, ns><<<blocks, threadsPerBlock, 0, stream>>>(
          d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, nf1, nf2, es_c, es_beta,
          sigma, d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  }

  return 0;
}

template<typename T, int ns>
static int cuspread2d_subprob(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan,
                       int blksize) {
  auto &stream = d_plan->stream;

  T es_c             = 4.0 / T(d_plan->spopts.nspread * d_plan->spopts.nspread);
  T es_beta          = d_plan->spopts.beta;
  int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;

  // assume that bin_size_x > ns/2;
  int bin_size_x = d_plan->opts.gpu_binsizex;
  int bin_size_y = d_plan->opts.gpu_binsizey;
  int numbins[2];
  numbins[0] = ceil((T)nf1 / bin_size_x);
  numbins[1] = ceil((T)nf2 / bin_size_y);

  T *d_kx               = d_plan->kxyz[0];
  T *d_ky               = d_plan->kxyz[1];
  cuda_complex<T> *d_c  = d_plan->c;
  cuda_complex<T> *d_fw = d_plan->fw;

  int *d_binsize         = d_plan->binsize;
  int *d_binstartpts     = d_plan->binstartpts;
  int *d_numsubprob      = d_plan->numsubprob;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_idxnupts        = d_plan->idxnupts;

  int totalnumsubprob   = d_plan->totalnumsubprob;
  int *d_subprob_to_bin = d_plan->subprob_to_bin;

  T sigma = d_plan->opts.upsampfac;

  const auto sharedplanorysize = shared_memory_required<T>(
      2, d_plan->spopts.nspread, d_plan->opts.gpu_binsizex, d_plan->opts.gpu_binsizey,
      d_plan->opts.gpu_binsizez, d_plan->opts.gpu_np);

  if (d_plan->opts.gpu_kerevalmeth) {
    if (const auto finufft_err =
            cufinufft_set_shared_memory(spread_2d_subprob<T, 1, ns>, 2, *d_plan) != 0) {
      return FINUFFT_ERR_INSUFFICIENT_SHMEM;
    }
    for (int t = 0; t < blksize; t++) {
      spread_2d_subprob<T, 1, ns><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, nf1, nf2, es_c, es_beta,
          sigma, d_binstartpts, d_binsize, bin_size_x, bin_size_y, d_subprob_to_bin,
          d_subprobstartpts, d_numsubprob, maxsubprobsize, numbins[0], numbins[1],
          d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  } else {
    if (const auto finufft_err =
            cufinufft_set_shared_memory(spread_2d_subprob<T, 0, ns>, 2, *d_plan) != 0) {
      return FINUFFT_ERR_INSUFFICIENT_SHMEM;
    }
    for (int t = 0; t < blksize; t++) {
      spread_2d_subprob<T, 0, ns><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_kx, d_ky, d_c + t * M, d_fw + t * nf1 * nf2, M, nf1, nf2, es_c, es_beta,
          sigma, d_binstartpts, d_binsize, bin_size_x, bin_size_y, d_subprob_to_bin,
          d_subprobstartpts, d_numsubprob, maxsubprobsize, numbins[0], numbins[1],
          d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  }
  return 0;
}

// Functor to handle function selection (nuptsdriven vs subprob)
struct Spread2DDispatcher {
  template<int ns, typename T>
  int operator()(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan,
                 int blksize) const {
    switch (d_plan->opts.gpu_method) {
    case 1:
      return cuspread2d_nuptsdriven<T, ns>(nf1, nf2, M, d_plan, blksize);
    case 2:
      return cuspread2d_subprob<T, ns>(nf1, nf2, M, d_plan, blksize);
    case 3:
      return cuspread2d_output_driven<T, ns>(nf1, nf2, M, d_plan, blksize);
    default:
      std::cerr << "[cuspread2d] error: incorrect method, should be 1, 2 or 3\n";
      return FINUFFT_ERR_METHOD_NOTVALID;
    }
  }
};

// Updated cuspread2d using generic dispatch
template<typename T> int cuspread2d(cufinufft_plan_t<T> *d_plan, int blksize) {
  /*
    A wrapper for different spreading methods.

    Methods available:
        (1) Non-uniform points driven
        (2) Subproblem

    Melody Shih 07/25/19

    Now the function is updated to dispatch based on ns. This is to avoid alloca which
    it seems slower according to the MRI community.
    Marco Barbone 01/30/25
  */
  return launch_dispatch_ns<Spread2DDispatcher, T>(
      Spread2DDispatcher(), d_plan->spopts.nspread, d_plan->nf123[0], d_plan->nf123[1], d_plan->M,
      d_plan, blksize);
}

template<typename T>
int cuspread2d_nuptsdriven_prop(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan) {
  auto &stream = d_plan->stream;

  if (d_plan->opts.gpu_sort) {
    int bin_size_x = d_plan->opts.gpu_binsizex;
    int bin_size_y = d_plan->opts.gpu_binsizey;
    if (bin_size_x < 0 || bin_size_y < 0) {
      std::cerr << "[cuspread2d_nuptsdriven_prop] error: invalid binsize "
                   "(binsizex, binsizey) = (";
      std::cerr << bin_size_x << "," << bin_size_y << ")" << std::endl;
      return FINUFFT_ERR_BINSIZE_NOTVALID;
    }

    int numbins[2];
    numbins[0] = ceil((T)nf1 / bin_size_x);
    numbins[1] = ceil((T)nf2 / bin_size_y);

    T *d_kx = d_plan->kxyz[0];
    T *d_ky = d_plan->kxyz[1];

    int *d_binsize     = d_plan->binsize;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_sortidx     = d_plan->sortidx;
    int *d_idxnupts    = d_plan->idxnupts;

    checkCudaErrors(cudaMemsetAsync(
             d_binsize, 0, numbins[0] * numbins[1] * sizeof(int), stream));

    calc_bin_size_noghost_2d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
        M, nf1, nf2, bin_size_x, bin_size_y, numbins[0], numbins[1], d_binsize, d_kx,
        d_ky, d_sortidx);
    RETURN_IF_CUDA_ERROR

    int n = numbins[0] * numbins[1];
    thrust::device_ptr<int> d_ptr(d_binsize);
    thrust::device_ptr<int> d_result(d_binstartpts);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);

    calc_inverse_of_global_sort_index_2d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
        M, bin_size_x, bin_size_y, numbins[0], numbins[1], d_binstartpts, d_sortidx, d_kx,
        d_ky, d_idxnupts, nf1, nf2);
    RETURN_IF_CUDA_ERROR
  } else {
    int *d_idxnupts = d_plan->idxnupts;
    thrust::sequence(thrust::cuda::par.on(stream), d_idxnupts, d_idxnupts + M);
    RETURN_IF_CUDA_ERROR
  }

  return 0;
}

template<typename T>
int cuspread2d_subprob_prop(int nf1, int nf2, int M, cufinufft_plan_t<T> *d_plan)
/*
    This function determines the properties for spreading that are independent
    of the strength of the nodes,  only relates to the locations of the nodes,
    which only needs to be done once.
*/
{
  auto &stream = d_plan->stream;

  int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;
  int bin_size_x     = d_plan->opts.gpu_binsizex;
  int bin_size_y     = d_plan->opts.gpu_binsizey;
  if (bin_size_x < 0 || bin_size_y < 0) {
    std::cerr << "[cuspread2d_subprob_prop] error: invalid binsize (binsizex, "
                 "binsizey) = (";
    std::cerr << bin_size_x << "," << bin_size_y << ")" << std::endl;
    return FINUFFT_ERR_BINSIZE_NOTVALID;
  }
  int numbins[2];
  numbins[0] = ceil((T)nf1 / bin_size_x);
  numbins[1] = ceil((T)nf2 / bin_size_y);

  T *d_kx = d_plan->kxyz[0];
  T *d_ky = d_plan->kxyz[1];

  int *d_binsize         = d_plan->binsize;
  int *d_binstartpts     = d_plan->binstartpts;
  int *d_sortidx         = d_plan->sortidx;
  int *d_numsubprob      = d_plan->numsubprob;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_idxnupts        = d_plan->idxnupts;

  int *d_subprob_to_bin = NULL;

  checkCudaErrors(
           cudaMemsetAsync(d_binsize, 0, numbins[0] * numbins[1] * sizeof(int), stream));

  calc_bin_size_noghost_2d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, nf1, nf2, bin_size_x, bin_size_y, numbins[0], numbins[1], d_binsize, d_kx, d_ky,
      d_sortidx);
  RETURN_IF_CUDA_ERROR

  int n = numbins[0] * numbins[1];
  thrust::device_ptr<int> d_ptr(d_binsize);
  thrust::device_ptr<int> d_result(d_binstartpts);
  thrust::exclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);

  calc_inverse_of_global_sort_index_2d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, bin_size_x, bin_size_y, numbins[0], numbins[1], d_binstartpts, d_sortidx, d_kx,
      d_ky, d_idxnupts, nf1, nf2);
  RETURN_IF_CUDA_ERROR
  calc_subprob_2d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      d_binsize, d_numsubprob, maxsubprobsize, numbins[0] * numbins[1]);
  RETURN_IF_CUDA_ERROR

  d_ptr    = thrust::device_pointer_cast(d_numsubprob);
  d_result = thrust::device_pointer_cast(d_subprobstartpts + 1);
  thrust::inclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);

  checkCudaErrors(cudaMemsetAsync(d_subprobstartpts, 0, sizeof(int), stream));

  int totalnumsubprob;
  checkCudaErrors(cudaMemcpyAsync(&totalnumsubprob, &d_subprobstartpts[n],
                                           sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
  checkCudaErrors(
           cudaMallocWrapper(&d_subprob_to_bin, totalnumsubprob * sizeof(int), stream,
                             d_plan->supports_pools));
  map_b_into_subprob_2d<<<(numbins[0] * numbins[1] + 1024 - 1) / 1024, 1024, 0, stream>>>(
      d_subprob_to_bin, d_subprobstartpts, d_numsubprob, numbins[0] * numbins[1]);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[%s] Error: %s\n", __func__, cudaGetErrorString(err));
    cudaFree(d_subprob_to_bin);
    return FINUFFT_ERR_CUDA_FAILURE;
  }

  assert(d_subprob_to_bin != NULL);
  cudaFreeWrapper(d_plan->subprob_to_bin, stream, d_plan->supports_pools);
  d_plan->subprob_to_bin  = d_subprob_to_bin;
  d_plan->totalnumsubprob = totalnumsubprob;

  return 0;
}

template int cuspread2d<float>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cuspread2d<double>(cufinufft_plan_t<double> *d_plan, int blksize);
template int cuspread2d_subprob_prop<float>(int nf1, int nf2, int M,
                                            cufinufft_plan_t<float> *d_plan);
template int cuspread2d_subprob_prop<double>(int nf1, int nf2, int M,
                                             cufinufft_plan_t<double> *d_plan);
template int cuspread2d_nuptsdriven_prop<float>(int nf1, int nf2, int M,
                                                cufinufft_plan_t<float> *d_plan);
template int cuspread2d_nuptsdriven_prop<double>(int nf1, int nf2, int M,
                                                 cufinufft_plan_t<double> *d_plan);

} // namespace spreadinterp
} // namespace cufinufft
