#include <cassert>
#include <iostream>

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cuComplex.h>

#include <cufinufft/common.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/memtransfer.h>
#include <cufinufft/precision_independent.h>
#include <cufinufft/spreadinterp.h>

#include "spreadinterp1d.cuh"

using namespace cufinufft::common;
using namespace cufinufft::memtransfer;

namespace cufinufft {
namespace spreadinterp {

/* ------------------------ 1d Spreading Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */

template<typename T, int KEREVALMETH, int ns>
static __global__ void spread_1d_nuptsdriven(const T *x, const cuda_complex<T> *c,
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
static __global__ void calc_bin_size_noghost_1d(int M, int nf1, int bin_size_x, int nbinx,
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
static __global__ void calc_inverse_of_global_sort_idx_1d(
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
static __global__ void spread_1d_output_driven(
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

  using mdspan_t = mdspan<T, extents<int, dynamic_extent, ns>>;
  auto kerevals = mdspan_t((T *)sharedbuf, np);
  // sharedbuf + size of kerevals in bytes
  // Offset pointer into sharedbuf after kerevals
  // Create span using pointer + size

  auto nupts_sm = span(
      reinterpret_cast<cuda_complex<T> *>(kerevals.data_handle() + kerevals.size()),
      np);

  auto shift = span(reinterpret_cast<int *>(nupts_sm.data() + nupts_sm.size()), np);

  auto local_subgrid = span<cuda_complex<T>>(
      reinterpret_cast<cuda_complex<T> *>(shift.data() + shift.size()), padded_size_x);

  // set local_subgrid to zero
  for (int i = threadIdx.x; i < local_subgrid.size(); i += blockDim.x) {
    local_subgrid[i] = {0, 0};
  }
  __syncthreads();

  for (int batch_begin = 0; batch_begin < nupts; batch_begin += np) {
    const auto batch_size = min(np, nupts - batch_begin);
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
      const int nuptsidx = idxnupts[ptstart + i + batch_begin];
      // index of the current point within the batch
      const auto x_rescaled = fold_rescale(x[nuptsidx], nf1);
      nupts_sm[i]           = c[nuptsidx];
      auto [xstart, xend]   = interval(ns, x_rescaled);
      const T x1            = T(xstart) - x_rescaled;

      shift[i] = xstart - xoffset;

      if constexpr (KEREVALMETH == 1) {
        eval_kernel_vec_horner<T, ns>(&kerevals(i, 0), x1, sigma);
      } else {
        eval_kernel_vec<T, ns>(&kerevals(i, 0), x1, es_c, es_beta);
      }
    }
    __syncthreads();

    for (auto i = 0; i < batch_size; i++) {
      // strength from shared memory

      const auto cnow             = nupts_sm[i];
      const auto xstart           = shift[i];
      static constexpr auto total = ns;
      for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int ix = xstart + idx + ns_2;
        if constexpr (std::is_same_v<T, float>) {
          if (ix >= (padded_size_x) || ix < 0) break;
        }
        // separable window weights
        const auto kervalue = kerevals(i, idx);
        // accumulate
        const cuda_complex<T> res{cnow.x * kervalue, cnow.y * kervalue};
        local_subgrid[ix] += res;
      }
      __syncthreads();
    }
  }
  /* write to global memory */
  for (int k = threadIdx.x; k < local_subgrid.size(); k += blockDim.x) {
    auto ix = xoffset - ns_2 + k;
    if (ix < (nf1 + ns_2)) {
      ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      atomicAddComplexGlobal<T>(fw + ix, local_subgrid[k]);
    }
  }
}

template<typename T, int KEREVALMETH, int ns>
static __global__ void spread_1d_subprob(
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

// Functor to handle function selection (nuptsdriven vs subprob)
struct Spread1DDispatcher {
  template<int ns, typename T>
  int operator()(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize) const {
    switch (d_plan->opts.gpu_method) {
    case 1:
      return cuspread1d_nuptsdriven<T, ns>(nf1, M, d_plan, blksize);
    case 2:
      return cuspread1d_subprob<T, ns>(nf1, M, d_plan, blksize);
    case 3:
      return cuspread1d_output_driven<T, ns>(nf1, M, d_plan, blksize);
    default:
      std::cerr << "[cuspread1d] error: incorrect method, should be 1, 2 or 3\n";
      return FINUFFT_ERR_METHOD_NOTVALID;
    }
  }
};

// Updated cuspread1d using generic dispatch
template<typename T> int cuspread1d(cufinufft_plan_t<T> *d_plan, int blksize) {
  /*
    A wrapper for different spreading methods.

    Methods available:
        (1) Non-uniform points driven

    Melody Shih 11/21/21

    Now the function is updated to dispatch based on ns. This is to avoid alloca which
    it seems slower according to the MRI community.
    Marco Barbone 01/30/25
 */
  return launch_dispatch_ns<Spread1DDispatcher, T>(Spread1DDispatcher(),
                                                   d_plan->spopts.nspread, d_plan->nf123[0],
                                                   d_plan->M, d_plan, blksize);
}

template<typename T>
int cuspread1d_nuptsdriven_prop(int nf1, int M, cufinufft_plan_t<T> *d_plan) {
  auto &stream = d_plan->stream;
  if (d_plan->opts.gpu_sort) {
    int bin_size_x = d_plan->opts.gpu_binsizex;
    if (bin_size_x < 0) {
      std::cerr << "[cuspread1d_nuptsdriven_prop] error: invalid binsize (binsizex) = ("
                << bin_size_x << ")\n";
      return FINUFFT_ERR_BINSIZE_NOTVALID;
    }

    int numbins = ceil((T)nf1 / bin_size_x);

    T *d_kx = d_plan->kxyz[0];

    int *d_binsize     = d_plan->binsize;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_sortidx     = d_plan->sortidx;
    int *d_idxnupts    = d_plan->idxnupts;

    checkCudaErrors(
        cudaMemsetAsync(d_binsize, 0, numbins * sizeof(int), stream));
    calc_bin_size_noghost_1d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
        M, nf1, bin_size_x, numbins, d_binsize, d_kx, d_sortidx);
    RETURN_IF_CUDA_ERROR

    int n = numbins;
    thrust::device_ptr<int> d_ptr(d_binsize);
    thrust::device_ptr<int> d_result(d_binstartpts);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);
    RETURN_IF_CUDA_ERROR

    calc_inverse_of_global_sort_idx_1d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
        M, bin_size_x, numbins, d_binstartpts, d_sortidx, d_kx, d_idxnupts, nf1);
    RETURN_IF_CUDA_ERROR
  } else {
    int *d_idxnupts = d_plan->idxnupts;
    thrust::sequence(thrust::cuda::par.on(stream), d_idxnupts, d_idxnupts + M);
    RETURN_IF_CUDA_ERROR
  }
  return 0;
}

template<typename T, int ns>
int cuspread1d_nuptsdriven(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize) {
  auto &stream = d_plan->stream;
  dim3 threadsPerBlock;
  dim3 blocks;

  int *d_idxnupts = d_plan->idxnupts;
  T es_c          = 4.0 / T(d_plan->spopts.nspread * d_plan->spopts.nspread);
  T es_beta       = d_plan->spopts.beta;
  T sigma         = d_plan->spopts.upsampfac;

  T *d_kx               = d_plan->kxyz[0];
  cuda_complex<T> *d_c  = d_plan->c;
  cuda_complex<T> *d_fw = d_plan->fw;

  threadsPerBlock.x = 16;
  threadsPerBlock.y = 1;
  blocks.x          = (M + threadsPerBlock.x - 1) / threadsPerBlock.x;
  blocks.y          = 1;

  if (d_plan->opts.gpu_kerevalmeth) {
    for (int t = 0; t < blksize; t++) {
      spread_1d_nuptsdriven<T, 1, ns><<<blocks, threadsPerBlock, 0, stream>>>(
          d_kx, d_c + t * M, d_fw + t * nf1, M, nf1, es_c, es_beta, sigma, d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  } else {
    for (int t = 0; t < blksize; t++) {
      spread_1d_nuptsdriven<T, 0, ns><<<blocks, threadsPerBlock, 0, stream>>>(
          d_kx, d_c + t * M, d_fw + t * nf1, M, nf1, es_c, es_beta, sigma, d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  }
  return 0;
}

template<typename T, int ns>
int cuspread1d_output_driven(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize) {
  auto &stream       = d_plan->stream;
  T es_c             = 4.0 / T(d_plan->spopts.nspread * d_plan->spopts.nspread);
  T es_beta          = d_plan->spopts.beta;
  int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;

  // assume that bin_size_x > ns/2;
  int bin_size_x = d_plan->opts.gpu_binsizex;
  int numbins    = ceil((T)nf1 / bin_size_x);

  T *d_kx               = d_plan->kxyz[0];
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
      1, d_plan->spopts.nspread, d_plan->opts.gpu_binsizex, d_plan->opts.gpu_binsizey,
      d_plan->opts.gpu_binsizez, d_plan->opts.gpu_np);

  if (d_plan->opts.gpu_kerevalmeth) {
    if (const auto finufft_err =
            cufinufft_set_shared_memory(spread_1d_output_driven<T, 1, ns>, 1, *d_plan) !=
            0) {
      return FINUFFT_ERR_INSUFFICIENT_SHMEM;
    }
    for (int t = 0; t < blksize; t++) {
      spread_1d_output_driven<T, 1, ns>
          <<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
              d_kx, d_c + t * M, d_fw + t * nf1, M, nf1, es_c, es_beta, sigma,
              d_binstartpts, d_binsize, bin_size_x, d_subprob_to_bin, d_subprobstartpts,
              d_numsubprob, maxsubprobsize, numbins, d_idxnupts, d_plan->opts.gpu_np);
      RETURN_IF_CUDA_ERROR
    }
  } else {
    if (const auto finufft_err =
            cufinufft_set_shared_memory(spread_1d_output_driven<T, 0, ns>, 1, *d_plan) !=
            0) {
      return FINUFFT_ERR_INSUFFICIENT_SHMEM;
    }
    for (int t = 0; t < blksize; t++) {
      spread_1d_output_driven<T, 0, ns>
          <<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
              d_kx, d_c + t * M, d_fw + t * nf1, M, nf1, es_c, es_beta, sigma,
              d_binstartpts, d_binsize, bin_size_x, d_subprob_to_bin, d_subprobstartpts,
              d_numsubprob, maxsubprobsize, numbins, d_idxnupts, d_plan->opts.gpu_np);
      RETURN_IF_CUDA_ERROR
    }
  }
  return 0;
}

template<typename T>
int cuspread1d_subprob_prop(int nf1, int M, cufinufft_plan_t<T> *d_plan)
/*
    This function determines the properties for spreading that are independent
    of the strength of the nodes,  only relates to the locations of the nodes,
    which only needs to be done once.
*/
{

  const auto maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;
  const auto bin_size_x     = d_plan->opts.gpu_binsizex;
  if (bin_size_x < 0) {
    std::cerr << "[cuspread1d_subprob_prop] error: invalid binsize (binsizex) = ("
              << bin_size_x << ")\n";
    return FINUFFT_ERR_BINSIZE_NOTVALID;
  }

  const auto numbins           = (nf1 + bin_size_x - 1) / bin_size_x;
  const auto d_kx              = d_plan->kxyz[0];
  const auto d_binsize         = d_plan->binsize;
  const auto d_binstartpts     = d_plan->binstartpts;
  const auto d_sortidx         = d_plan->sortidx;
  const auto d_numsubprob      = d_plan->numsubprob;
  const auto d_subprobstartpts = d_plan->subprobstartpts;
  const auto d_idxnupts        = d_plan->idxnupts;
  const auto stream            = d_plan->stream;

  int *d_subprob_to_bin = nullptr;

  cudaMemsetAsync(d_binsize, 0, numbins * sizeof(int), stream);
  RETURN_IF_CUDA_ERROR
  calc_bin_size_noghost_1d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, nf1, bin_size_x, numbins, d_binsize, d_kx, d_sortidx);
  RETURN_IF_CUDA_ERROR

  int n = numbins;
  thrust::device_ptr<int> d_ptr(d_binsize);
  thrust::device_ptr<int> d_result(d_binstartpts);
  thrust::exclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);

  calc_inverse_of_global_sort_idx_1d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, bin_size_x, numbins, d_binstartpts, d_sortidx, d_kx, d_idxnupts, nf1);
  RETURN_IF_CUDA_ERROR

  calc_subprob_1d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(d_binsize, d_numsubprob,
                                                              maxsubprobsize, numbins);
  RETURN_IF_CUDA_ERROR

  d_ptr    = thrust::device_pointer_cast(d_numsubprob);
  d_result = thrust::device_pointer_cast(d_subprobstartpts + 1);
  thrust::inclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);
  RETURN_IF_CUDA_ERROR

  cudaMemsetAsync(d_subprobstartpts, 0, sizeof(int), stream);
  RETURN_IF_CUDA_ERROR

  int totalnumsubprob{};
  cudaMemcpyAsync(&totalnumsubprob, &d_subprobstartpts[n], sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  RETURN_IF_CUDA_ERROR

  cudaMallocWrapper(&d_subprob_to_bin, totalnumsubprob * sizeof(int), stream,
                    d_plan->supports_pools);
  RETURN_IF_CUDA_ERROR

  map_b_into_subprob_1d<<<(numbins + 1024 - 1) / 1024, 1024, 0, stream>>>(
      d_subprob_to_bin, d_subprobstartpts, d_numsubprob, numbins);
  RETURN_IF_CUDA_ERROR
  assert(d_subprob_to_bin != nullptr);
  cudaFreeWrapper(d_plan->subprob_to_bin, stream, d_plan->supports_pools);
  d_plan->subprob_to_bin  = d_subprob_to_bin;
  d_plan->totalnumsubprob = totalnumsubprob;

  return 0;
}

template<typename T, int ns>
int cuspread1d_subprob(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize) {
  auto &stream       = d_plan->stream;
  T es_c             = 4.0 / T(d_plan->spopts.nspread * d_plan->spopts.nspread);
  T es_beta          = d_plan->spopts.beta;
  int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;

  // assume that bin_size_x > ns/2;
  int bin_size_x = d_plan->opts.gpu_binsizex;
  int numbins    = ceil((T)nf1 / bin_size_x);

  T *d_kx               = d_plan->kxyz[0];
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
      1, d_plan->spopts.nspread, d_plan->opts.gpu_binsizex, d_plan->opts.gpu_binsizey,
      d_plan->opts.gpu_binsizez, d_plan->opts.gpu_np);

  if (d_plan->opts.gpu_kerevalmeth) {
    if (const auto finufft_err =
            cufinufft_set_shared_memory(spread_1d_subprob<T, 1, ns>, 1, *d_plan) != 0) {
      return FINUFFT_ERR_INSUFFICIENT_SHMEM;
    }
    for (int t = 0; t < blksize; t++) {
      spread_1d_subprob<T, 1, ns><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_kx, d_c + t * M, d_fw + t * nf1, M, nf1, es_c, es_beta, sigma, d_binstartpts,
          d_binsize, bin_size_x, d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
          maxsubprobsize, numbins, d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  } else {
    if (const auto finufft_err =
            cufinufft_set_shared_memory(spread_1d_subprob<T, 0, ns>, 1, *d_plan) != 0) {
      return FINUFFT_ERR_INSUFFICIENT_SHMEM;
    }
    for (int t = 0; t < blksize; t++) {
      spread_1d_subprob<T, 0, ns><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_kx, d_c + t * M, d_fw + t * nf1, M, nf1, es_c, es_beta, sigma, d_binstartpts,
          d_binsize, bin_size_x, d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
          maxsubprobsize, numbins, d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  }
  return 0;
}

template int cuspread1d<float>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cuspread1d<double>(cufinufft_plan_t<double> *d_plan, int blksize);
template int cuspread1d_nuptsdriven_prop<float>(int nf1, int M,
                                                cufinufft_plan_t<float> *d_plan);
template int cuspread1d_nuptsdriven_prop<double>(int nf1, int M,
                                                 cufinufft_plan_t<double> *d_plan);
template int cuspread1d_subprob_prop<float>(int nf1, int M,
                                            cufinufft_plan_t<float> *d_plan);
template int cuspread1d_subprob_prop<double>(int nf1, int M,
                                             cufinufft_plan_t<double> *d_plan);

} // namespace spreadinterp
} // namespace cufinufft
