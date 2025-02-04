#include <cassert>
#include <cufinufft/contrib/helper_cuda.h>
#include <iostream>

#include <cuComplex.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cufinufft/common.h>
#include <cufinufft/memtransfer.h>
#include <cufinufft/precision_independent.h>
#include <cufinufft/spreadinterp.h>

using namespace cufinufft::common;
using namespace cufinufft::memtransfer;

#include "spreadinterp1d.cuh"
#include <thrust/sort.h>

namespace cufinufft {
namespace spreadinterp {

// Functor to handle function selection (nuptsdriven vs subprob)
struct Spread1DDispatcher {
  template<int ns, typename T>
  int operator()(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize) const {
    switch (d_plan->opts.gpu_method) {
    case 1:
      return cuspread1d_nuptsdriven<T, ns>(nf1, M, d_plan, blksize);
    case 2:
      return cuspread1d_subprob<T, ns>(nf1, M, d_plan, blksize);
    default:
      std::cerr << "[cuspread1d] error: incorrect method, should be 1 or 2\n";
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
                                                   d_plan->spopts.nspread, d_plan->nf1,
                                                   d_plan->M, d_plan, blksize);
}

template<typename T> struct cmp : public thrust::binary_function<int, int, bool> {

  cmp(const T *kx) : kx(kx) {}

  __host__ __device__ bool operator()(const int a, const int b) const {
    return fold_rescale(kx[a], 1) < fold_rescale(kx[b], 1);
  }

private:
  const T *kx;
};

template<typename T>
int cuspread1d_nuptsdriven_prop(int nf1, int M, cufinufft_plan_t<T> *d_plan) {
  auto &stream = d_plan->stream;
  if (d_plan->opts.gpu_sort && d_plan->opts.gpu_method == 1) {
    int *d_idxnupts = d_plan->idxnupts;
    thrust::sequence(thrust::cuda::par.on(stream), d_idxnupts, d_idxnupts + M);
    RETURN_IF_CUDA_ERROR
    thrust::sort(thrust::cuda::par.on(stream), d_idxnupts, d_idxnupts + M,
                 cmp{d_plan->kx});
    RETURN_IF_CUDA_ERROR
    return 0;
  }
  if (d_plan->opts.gpu_sort) {
    int bin_size_x = d_plan->opts.gpu_binsizex;
    if (bin_size_x < 0) {
      std::cerr << "[cuspread1d_nuptsdriven_prop] error: invalid binsize (binsizex) = ("
                << bin_size_x << ")\n";
      return FINUFFT_ERR_BINSIZE_NOTVALID;
    }

    int numbins = ceil((T)nf1 / bin_size_x);

    T *d_kx = d_plan->kx;

    int *d_binsize     = d_plan->binsize;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_sortidx     = d_plan->sortidx;
    int *d_idxnupts    = d_plan->idxnupts;

    int ier;
    if ((ier = checkCudaErrors(
             cudaMemsetAsync(d_binsize, 0, numbins * sizeof(int), stream))))
      return ier;
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
  T es_c          = d_plan->spopts.ES_c;
  T es_beta       = d_plan->spopts.ES_beta;
  T sigma         = d_plan->spopts.upsampfac;

  T *d_kx               = d_plan->kx;
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
  const auto d_kx              = d_plan->kx;
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
  T es_c             = d_plan->spopts.ES_c;
  T es_beta          = d_plan->spopts.ES_beta;
  int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;

  // assume that bin_size_x > ns/2;
  int bin_size_x = d_plan->opts.gpu_binsizex;
  int numbins    = ceil((T)nf1 / bin_size_x);

  T *d_kx               = d_plan->kx;
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

  const auto sharedplanorysize =
      shared_memory_required<T>(1, d_plan->spopts.nspread, d_plan->opts.gpu_binsizex,
                                d_plan->opts.gpu_binsizey, d_plan->opts.gpu_binsizez);

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
