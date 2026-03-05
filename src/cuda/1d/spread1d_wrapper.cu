#include <cassert>
#include <cmath>
#include <cuda/std/mdspan>
#include <iostream>

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cuComplex.h>

#include <cufinufft/common.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/contrib/helper_math.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/utils.h>
#include <cufinufft/common_kernels.hpp>

using namespace cufinufft::common;
using namespace cufinufft::utils;

namespace cufinufft {
namespace spreadinterp {

using cuda::std::dextents;
using cuda::std::dynamic_extent;
using cuda::std::extents;
using cuda::std::mdspan;
using cuda::std::span;

/* ------------------------ 1d Spreading Kernels ----------------------------*/

static __global__ void calc_subprob_1d(const int *bin_size, int *num_subprob,
                                       int maxsubprobsize, int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    num_subprob[i] = ceil(bin_size[i] / (float)maxsubprobsize);
  }
}

static __global__ void map_b_into_subprob_1d(int *d_subprob_to_bin,
                                             const int *d_subprobstartpts,
                                             const int *d_numsubprob, int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    for (int j = 0; j < d_numsubprob[i]; j++) {
      d_subprob_to_bin[d_subprobstartpts[i] + j] = i;
    }
  }
}

/* Kernels for SubProb Method */
template<typename T, int KEREVALMETH, int ns>
static __global__ void spread_1d_output_driven(
    const T *x, const cuda_complex<T> *c, cuda_complex<T> *fw, int M, int nf1, T es_c,
    T es_beta, T sigma, const int *binstartpts, const int *bin_size, int bin_size_x,
    const int *subprob_to_bin, const int *subprobstartpts, const int *numsubprob,
    int maxsubprobsize, int nbinx, const int *idxnupts, const int np) {
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
  auto kerevals  = mdspan_t((T *)sharedbuf, np);
  // sharedbuf + size of kerevals in bytes
  // Offset pointer into sharedbuf after kerevals
  // Create span using pointer + size

  auto nupts_sm = span(
      reinterpret_cast<cuda_complex<T> *>(kerevals.data_handle() + kerevals.size()), np);

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

template<typename T, int ns>
static void cuspread1d_output_driven(int nf1, int M, const cufinufft_plan_t<T> &d_plan,
                                     int blksize) {
  auto &stream       = d_plan.stream;
  T es_c             = 4.0 / T(d_plan.spopts.nspread * d_plan.spopts.nspread);
  T es_beta          = d_plan.spopts.beta;
  int maxsubprobsize = d_plan.opts.gpu_maxsubprobsize;

  // assume that bin_size_x > ns/2;
  int bin_size_x = d_plan.opts.gpu_binsizex;
  int numbins    = ceil((T)nf1 / bin_size_x);

  const T *d_kx              = d_plan.kxyz[0];
  const cuda_complex<T> *d_c = d_plan.c;
  cuda_complex<T> *d_fw      = d_plan.fw;

  const int *d_binsize         = dethrust(d_plan.binsize);
  const int *d_binstartpts     = dethrust(d_plan.binstartpts);
  const int *d_numsubprob      = dethrust(d_plan.numsubprob);
  const int *d_subprobstartpts = dethrust(d_plan.subprobstartpts);
  const int *d_idxnupts        = dethrust(d_plan.idxnupts);

  int totalnumsubprob         = d_plan.totalnumsubprob;
  const int *d_subprob_to_bin = dethrust(d_plan.subprob_to_bin);

  T sigma = d_plan.opts.upsampfac;

  const auto sharedplanorysize = shared_memory_required<T>(
      1, d_plan.spopts.nspread, d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey,
      d_plan.opts.gpu_binsizez, d_plan.opts.gpu_np);

  if (d_plan.opts.gpu_kerevalmeth) {
    cufinufft_set_shared_memory(spread_1d_output_driven<T, 1, ns>, 1, d_plan);
    for (int t = 0; t < blksize; t++) {
      spread_1d_output_driven<T, 1, ns>
          <<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
              d_kx, d_c + t * M, d_fw + t * nf1, M, nf1, es_c, es_beta, sigma,
              d_binstartpts, d_binsize, bin_size_x, d_subprob_to_bin, d_subprobstartpts,
              d_numsubprob, maxsubprobsize, numbins, d_idxnupts, d_plan.opts.gpu_np);
      THROW_IF_CUDA_ERROR
    }
  } else {
    cufinufft_set_shared_memory(spread_1d_output_driven<T, 0, ns>, 1, d_plan);
    for (int t = 0; t < blksize; t++) {
      spread_1d_output_driven<T, 0, ns>
          <<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
              d_kx, d_c + t * M, d_fw + t * nf1, M, nf1, es_c, es_beta, sigma,
              d_binstartpts, d_binsize, bin_size_x, d_subprob_to_bin, d_subprobstartpts,
              d_numsubprob, maxsubprobsize, numbins, d_idxnupts, d_plan.opts.gpu_np);
      THROW_IF_CUDA_ERROR
    }
  }
}

template<typename T>
static void cuspread1d_subprob_prop(cufinufft_plan_t<T> &d_plan)
/*
    This function determines the properties for spreading that are independent
    of the strength of the nodes,  only relates to the locations of the nodes,
    which only needs to be done once.
*/
{
  int M   = d_plan.M;
  int nf1 = d_plan.nf123[0];

  const auto maxsubprobsize = d_plan.opts.gpu_maxsubprobsize;
  const auto bin_size_x     = d_plan.opts.gpu_binsizex;
  if (bin_size_x < 0) {
    std::cerr << "[cuspread1d_subprob_prop] error: invalid binsize (binsizex) = ("
              << bin_size_x << ")\n";
    throw int(FINUFFT_ERR_BINSIZE_NOTVALID);
  }

  const auto numbins           = (nf1 + bin_size_x - 1) / bin_size_x;
  const auto d_kx              = d_plan.kxyz[0];
  const auto d_binsize         = dethrust(d_plan.binsize);
  const auto d_binstartpts     = dethrust(d_plan.binstartpts);
  const auto d_sortidx         = dethrust(d_plan.sortidx);
  const auto d_numsubprob      = dethrust(d_plan.numsubprob);
  const auto d_subprobstartpts = dethrust(d_plan.subprobstartpts);
  const auto d_idxnupts        = dethrust(d_plan.idxnupts);
  const auto stream            = d_plan.stream;

  cudaMemsetAsync(d_binsize, 0, numbins * sizeof(int), stream);
  THROW_IF_CUDA_ERROR
  calc_bin_size_noghost<T,1><<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, d_plan.nf123, {bin_size_x,1,1}, {numbins,1,1}, d_binsize, d_plan.kxyz, d_sortidx);
  THROW_IF_CUDA_ERROR

  int n = numbins;
  thrust::device_ptr<int> d_ptr(d_binsize);
  thrust::device_ptr<int> d_result(d_binstartpts);
  thrust::exclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);

  calc_inverse_of_global_sort_idx<T,1><<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, {bin_size_x,1,1}, {numbins,1,1}, d_binstartpts, d_sortidx, d_plan.kxyz, d_idxnupts, d_plan.nf123);
  THROW_IF_CUDA_ERROR

  calc_subprob_1d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(d_binsize, d_numsubprob,
                                                              maxsubprobsize, numbins);
  THROW_IF_CUDA_ERROR

  d_ptr    = thrust::device_pointer_cast(d_numsubprob);
  d_result = thrust::device_pointer_cast(d_subprobstartpts + 1);
  thrust::inclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);
  THROW_IF_CUDA_ERROR

  cudaMemsetAsync(d_subprobstartpts, 0, sizeof(int), stream);
  THROW_IF_CUDA_ERROR

  int totalnumsubprob{};
  cudaMemcpyAsync(&totalnumsubprob, &d_subprobstartpts[n], sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  THROW_IF_CUDA_ERROR

  gpu_array<int> d_subprob_to_bin(totalnumsubprob, d_plan.alloc);

  map_b_into_subprob_1d<<<(numbins + 1024 - 1) / 1024, 1024, 0, stream>>>(
      dethrust(d_subprob_to_bin), d_subprobstartpts, d_numsubprob, numbins);
  THROW_IF_CUDA_ERROR
  d_plan.subprob_to_bin.clear();
  d_subprob_to_bin.swap(d_plan.subprob_to_bin);
  d_plan.totalnumsubprob = totalnumsubprob;
}

// Functor to handle function selection (nuptsdriven vs subprob)
struct Spread1DDispatcher {
  template<int ns, typename T>
  void operator()(int nf1, int M, const cufinufft_plan_t<T> &d_plan, int blksize) const {
    switch (d_plan.opts.gpu_method) {
    case 1:
      return cuspread_nupts_driven<T, 1, ns>(d_plan, blksize);
    case 2:
      return cuspread_subprob<T, 1, ns>(d_plan, blksize);
    case 3:
      return cuspread1d_output_driven<T, ns>(nf1, M, d_plan, blksize);
    default:
      std::cerr << "[cuspread1d] error: incorrect method, should be 1, 2 or 3\n";
      throw int(FINUFFT_ERR_METHOD_NOTVALID);
    }
  }
};

// Updated cuspread1d using generic dispatch
template<typename T> void cuspread1d(const cufinufft_plan_t<T> &d_plan, int blksize) {
  /*
    A wrapper for different spreading methods.

    Methods available:
        (1) Non-uniform points driven

    Melody Shih 11/21/21

    Now the function is updated to dispatch based on ns. This is to avoid alloca which
    it seems slower according to the MRI community.
    Marco Barbone 01/30/25
 */
  launch_dispatch_ns<Spread1DDispatcher, T>(Spread1DDispatcher(), d_plan.spopts.nspread,
                                            d_plan.nf123[0], d_plan.M, d_plan, blksize);
}
template void cuspread1d<float>(const cufinufft_plan_t<float> &d_plan, int blksize);
template void cuspread1d<double>(const cufinufft_plan_t<double> &d_plan, int blksize);

template<typename T> void cuspread1d_prop(cufinufft_plan_t<T> &d_plan) {
  if (d_plan.opts.gpu_method == 1) cuspread_nuptsdriven_prop<T,1>(d_plan);
  if (d_plan.opts.gpu_method == 2) cuspread1d_subprob_prop(d_plan);
  if (d_plan.opts.gpu_method == 3) cuspread1d_subprob_prop(d_plan);
}
template void cuspread1d_prop(cufinufft_plan_t<float> &d_plan);
template void cuspread1d_prop(cufinufft_plan_t<double> &d_plan);

} // namespace spreadinterp
} // namespace cufinufft
